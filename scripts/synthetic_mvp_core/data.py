from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


@dataclass
class TaskSpec:
    num_values: int = 4
    pad_token: int = 0
    sep_token: int | None = None
    eos_token: int | None = None

    def __post_init__(self) -> None:
        if self.sep_token is None:
            self.sep_token = self.num_values + 1
        if self.eos_token is None:
            self.eos_token = self.num_values + 2
        expected = {
            "pad_token": 0,
            "sep_token": self.num_values + 1,
            "eos_token": self.num_values + 2,
        }
        actual = {
            "pad_token": self.pad_token,
            "sep_token": self.sep_token,
            "eos_token": self.eos_token,
        }
        if actual != expected:
            raise ValueError(f"unsupported special token layout: expected {expected}, got {actual}")

    @property
    def vocab_size(self) -> int:
        return int(self.eos_token) + 1

@dataclass
class CopyBatch:
    tokens: torch.Tensor
    loss_positions: torch.Tensor
    targets: torch.Tensor
    N: int
    T_raw: int
    T: int

def padded_copy_lengths(N: int, block_size: int) -> tuple[int, int]:
    T_raw = 2 * int(N) + 2
    T = int(math.ceil(T_raw / block_size) * block_size)
    return T_raw, T

def make_full_copy_batch(
    batch_size: int,
    N: int,
    block_size: int,
    spec: TaskSpec,
    device: torch.device,
    seed: int,
    batch_index: int,
    stream: str = "train",
) -> CopyBatch:
    T_raw, T = padded_copy_lengths(N, block_size)
    derived_seed = stable_int_seed(seed, batch_index, N, spec.num_values, stream)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(derived_seed)
    source = torch.randint(
        1,
        spec.num_values + 1,
        (batch_size, N),
        dtype=torch.long,
        generator=gen,
    ).to(device)
    tokens = torch.full((batch_size, T), spec.pad_token, dtype=torch.long, device=device)
    tokens[:, :N] = source
    tokens[:, N] = int(spec.sep_token)
    tokens[:, N + 1 : 2 * N + 1] = source
    tokens[:, 2 * N + 1] = int(spec.eos_token)
    loss_positions = torch.arange(N, 2 * N + 1, dtype=torch.long, device=device)
    targets = tokens[:, loss_positions + 1]
    return CopyBatch(
        tokens=tokens,
        loss_positions=loss_positions,
        targets=targets,
        N=N,
        T_raw=T_raw,
        T=T,
    )

def stable_int_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)

def canonical_task_name(task: str) -> str:
    if task != "copy":
        raise ValueError(f"v0.6 only supports task=copy, got {task}")
    return "copy"

def make_batch(
    args,
    spec: TaskSpec,
    device: torch.device,
    seq_len: int | None = None,
    batch_size: int | None = None,
    batch_index: int | None = None,
    stream: str = "train",
) -> CopyBatch:
    task = canonical_task_name(args.task)
    N = int(seq_len if seq_len is not None else args.seq_len)
    batch_size = int(batch_size if batch_size is not None else args.batch_size)
    if task == "copy":
        if batch_index is None:
            raise ValueError("copy batch generation requires an explicit batch_index")
        return make_full_copy_batch(
            batch_size=batch_size,
            N=N,
            block_size=args.block_size,
            spec=spec,
            device=device,
            seed=int(getattr(args, "seed")),
            batch_index=int(batch_index),
            stream=stream,
        )
    raise ValueError(f"unknown task: {args.task}")
