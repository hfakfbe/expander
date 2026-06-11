import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import random
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_structures import (
    DEFAULT_GRAPH_CONFIG,
    build_attention_mask,
    build_cross_mask,
    build_h_graph,
    build_local_mask,
    build_random_cross,
    build_random_cross_edges,
    build_zigzag_cross,
    build_zigzag_cross_edges,
    expected_raw_k,
    h_neighbors,
    mask_metrics,
    rot_g,
    rot_g_cyclic,
    validate_graph_config,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_neighbors(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = mask.sum(dim=-1)
    max_len = int(lengths.max().item())
    neighbors = torch.zeros((mask.shape[0], max_len), dtype=torch.long, device=mask.device)
    valid = torch.zeros((mask.shape[0], max_len), dtype=torch.bool, device=mask.device)
    for row in range(mask.shape[0]):
        idx = torch.nonzero(mask[row], as_tuple=False).flatten()
        neighbors[row, : len(idx)] = idx
        valid[row, : len(idx)] = True
    return neighbors, valid


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def neighbor_attention_from_table(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    gathered_k = k[:, :, neighbors.reshape(-1), :].reshape(
        batch, heads, seq_len, neighbors.shape[1], head_dim
    )
    gathered_v = v[:, :, neighbors.reshape(-1), :].reshape(
        batch, heads, seq_len, neighbors.shape[1], head_dim
    )
    scores = (q[:, :, :, None, :] * gathered_k).sum(dim=-1) * (head_dim ** -0.5)
    scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return (weights[..., None] * gathered_v).sum(dim=-2)


def neighbor_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neighbors, valid = mask_to_neighbors(mask)
    return neighbor_attention_from_table(q, k, v, neighbors, valid)


def cross_neighbors_to_block_pair_index(
    cross_neighbors: torch.Tensor,
    valid_cross_neighbors: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    records: list[list[int]] = []
    seq_len, max_cross = cross_neighbors.shape
    for src in range(seq_len):
        for slot in range(max_cross):
            if not bool(valid_cross_neighbors[src, slot].item()):
                continue
            dst = int(cross_neighbors[src, slot].item())
            records.append(
                [
                    src // block_size,
                    dst // block_size,
                    src % block_size,
                    dst % block_size,
                    src,
                    dst,
                    slot,
                ]
            )
    if not records:
        return torch.empty((0, 7), dtype=torch.long, device=cross_neighbors.device)
    out = torch.tensor(records, dtype=torch.long, device=cross_neighbors.device)
    order = (
        out[:, 0] * 10_000_000
        + out[:, 1] * 100_000
        + out[:, 2] * 1_000
        + out[:, 3]
    )
    return out[torch.argsort(order)]


def local_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    cross_neighbors: torch.Tensor | None,
    valid_cross_neighbors: torch.Tensor | None,
    dropout: nn.Module | None = None,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size for split attention")
    num_blocks = seq_len // block_size
    scale = head_dim ** -0.5

    q_blocks = q.view(batch, heads, num_blocks, block_size, head_dim)
    k_blocks = k.view(batch, heads, num_blocks, block_size, head_dim)
    v_blocks = v.view(batch, heads, num_blocks, block_size, head_dim)
    local_scores = torch.einsum("bhqtd,bhqsd->bhqts", q_blocks, k_blocks) * scale
    local_scores_flat = local_scores.reshape(batch, heads, seq_len, block_size)

    has_cross = (
        cross_neighbors is not None
        and valid_cross_neighbors is not None
        and cross_neighbors.shape[1] > 0
    )
    if has_cross:
        gathered_k = k[:, :, cross_neighbors.reshape(-1), :].reshape(
            batch, heads, seq_len, cross_neighbors.shape[1], head_dim
        )
        cross_scores = (q[:, :, :, None, :] * gathered_k).sum(dim=-1) * scale
        cross_scores = cross_scores.masked_fill(
            ~valid_cross_neighbors[None, None, :, :], torch.finfo(cross_scores.dtype).min
        )
        scores = torch.cat([local_scores_flat, cross_scores], dim=-1)
    else:
        scores = local_scores_flat

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    local_weights = weights[..., :block_size].reshape(batch, heads, num_blocks, block_size, block_size)
    local_out = torch.einsum("bhqts,bhqsd->bhqtd", local_weights, v_blocks).reshape(
        batch, heads, seq_len, head_dim
    )

    if not has_cross:
        return local_out
    gathered_v = v[:, :, cross_neighbors.reshape(-1), :].reshape(
        batch, heads, seq_len, cross_neighbors.shape[1], head_dim
    )
    cross_weights = weights[..., block_size:]
    cross_out = (cross_weights[..., None] * gathered_v).sum(dim=-2)
    return local_out + cross_out


def local_blockpair_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    cross_neighbors: torch.Tensor | None,
    valid_cross_neighbors: torch.Tensor | None,
    block_pair_index: torch.Tensor | None,
    dropout: nn.Module | None = None,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size for block-pair attention")
    num_blocks = seq_len // block_size
    scale = head_dim ** -0.5

    q_blocks = q.view(batch, heads, num_blocks, block_size, head_dim)
    k_blocks = k.view(batch, heads, num_blocks, block_size, head_dim)
    v_blocks = v.view(batch, heads, num_blocks, block_size, head_dim)
    local_scores = torch.einsum("bhqtd,bhqsd->bhqts", q_blocks, k_blocks) * scale
    local_scores_flat = local_scores.reshape(batch, heads, seq_len, block_size)

    has_cross = (
        cross_neighbors is not None
        and valid_cross_neighbors is not None
        and block_pair_index is not None
        and cross_neighbors.shape[1] > 0
        and block_pair_index.numel() > 0
    )
    if has_cross:
        max_cross = cross_neighbors.shape[1]
        src_tokens = block_pair_index[:, 4]
        dst_tokens = block_pair_index[:, 5]
        slots = block_pair_index[:, 6]
        cross_scores = torch.full(
            (batch, heads, seq_len, max_cross),
            torch.finfo(q.dtype).min,
            dtype=q.dtype,
            device=q.device,
        )
        q_edges = q[:, :, src_tokens, :]
        k_edges = k[:, :, dst_tokens, :]
        edge_scores = (q_edges * k_edges).sum(dim=-1) * scale
        cross_scores[:, :, src_tokens, slots] = edge_scores
        scores = torch.cat([local_scores_flat, cross_scores], dim=-1)
    else:
        scores = local_scores_flat

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    local_weights = weights[..., :block_size].reshape(batch, heads, num_blocks, block_size, block_size)
    local_out = torch.einsum("bhqts,bhqsd->bhqtd", local_weights, v_blocks).reshape(
        batch, heads, seq_len, head_dim
    )

    if not has_cross:
        return local_out

    cross_weights = weights[..., block_size:]
    cross_out = q.new_zeros((batch, heads, seq_len, head_dim))
    src_tokens = block_pair_index[:, 4]
    dst_tokens = block_pair_index[:, 5]
    slots = block_pair_index[:, 6]
    edge_weights = cross_weights[:, :, src_tokens, slots]
    edge_values = v[:, :, dst_tokens, :]
    edge_out = edge_weights[..., None] * edge_values
    flat_out = cross_out.reshape(batch * heads, seq_len, head_dim)
    flat_edges = edge_out.reshape(batch * heads, src_tokens.numel(), head_dim)
    flat_out.index_add_(1, src_tokens, flat_edges)
    return local_out + cross_out


def run_mask_tests(device: torch.device) -> list[dict]:
    results = []
    for seq_len in (64, 128):
        for block_size in (8, 16):
            if seq_len % block_size != 0:
                continue
            for degree in (2, 3):
                if degree >= block_size:
                    continue
                num_blocks = seq_len // block_size
                for v in range(num_blocks):
                    for port in range(block_size):
                        w, j = rot_g_cyclic(v, port, num_blocks, block_size)
                        vv, ii = rot_g_cyclic(w, j, num_blocks, block_size)
                        assert (vv, ii) == (v, port)
                        assert len(h_neighbors(port, block_size, degree)) == degree
                mask = build_attention_mask("zigzag", seq_len, block_size, degree, device, seed=0)
                assert mask.shape == (seq_len, seq_len)
                assert bool(mask.any(dim=-1).all())
                assert int(torch.nonzero(mask, as_tuple=False).min().item()) >= 0
                q = torch.randn(2, 2, seq_len, 8, device=device)
                k = torch.randn(2, 2, seq_len, 8, device=device)
                v = torch.randn(2, 2, seq_len, 8, device=device)
                dense = dense_attention(q, k, v, mask)
                neigh = neighbor_attention(q, k, v, mask)
                cross = build_cross_mask("zigzag", seq_len, block_size, degree, device, seed=0)
                cross_neighbors, valid_cross_neighbors = mask_to_neighbors(cross)
                split = local_cross_attention(
                    q, k, v, block_size, cross_neighbors, valid_cross_neighbors
                )
                block_pair_index = cross_neighbors_to_block_pair_index(
                    cross_neighbors, valid_cross_neighbors, block_size
                )
                blockpair = local_blockpair_attention(
                    q, k, v, block_size, cross_neighbors, valid_cross_neighbors, block_pair_index
                )
                max_error = float((dense - neigh).abs().max().item())
                split_max_error = float((dense - split).abs().max().item())
                blockpair_max_error = float((dense - blockpair).abs().max().item())
                assert max_error < 1e-5, max_error
                assert split_max_error < 1e-5, split_max_error
                assert blockpair_max_error < 1e-5, blockpair_max_error
                row_metric = mask_metrics(mask, "zigzag", block_size, degree)
                row_metric.update(
                    {
                        "seq_len": seq_len,
                        "block_size": block_size,
                        "degree": degree,
                        "dense_neighbor_max_error": max_error,
                        "dense_split_max_error": split_max_error,
                        "dense_blockpair_max_error": blockpair_max_error,
                    }
                )
                results.append(row_metric)
    return results


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        block_pair_index: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_backend == "dense_mask":
            scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)
        elif self.attention_backend == "neighbor":
            if neighbors is None or valid_neighbors is None:
                raise ValueError("neighbor backend requires precomputed neighbor tables")
            batch, heads, seq_len, head_dim = q.shape
            gathered_k = k[:, :, neighbors.reshape(-1), :].reshape(
                batch, heads, seq_len, neighbors.shape[1], head_dim
            )
            gathered_v = v[:, :, neighbors.reshape(-1), :].reshape(
                batch, heads, seq_len, neighbors.shape[1], head_dim
            )
            scores = (q[:, :, :, None, :] * gathered_k).sum(dim=-1) * (head_dim ** -0.5)
            scores = scores.masked_fill(
                ~valid_neighbors[None, None, :, :], torch.finfo(scores.dtype).min
            )
            attn = self.dropout(torch.softmax(scores, dim=-1))
            out = (attn[..., None] * gathered_v).sum(dim=-2)
        elif self.attention_backend == "split":
            out = local_cross_attention(
                q, k, v, self.block_size, neighbors, valid_neighbors, self.dropout
            )
        elif self.attention_backend == "blockpair":
            out = local_blockpair_attention(
                q,
                k,
                v,
                self.block_size,
                neighbors,
                valid_neighbors,
                block_pair_index,
                self.dropout,
            )
        else:
            raise ValueError(f"unknown attention backend: {self.attention_backend}")
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(out)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedSelfAttention(d_model, num_heads, dropout, attention_backend, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        block_pair_index: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, neighbors, valid_neighbors, block_pair_index)
        return x + self.ffn(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(d_model, num_heads, ffn_dim, dropout, attention_backend, block_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        valid_neighbors: torch.Tensor | None = None,
        block_pair_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.token(x) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(h, mask, neighbors, valid_neighbors, block_pair_index)
        return self.head(self.norm(h[:, -1]))


@dataclass
class TaskSpec:
    num_keys: int = 64
    num_values: int = 10
    query_token: int = 75
    pad_token: int = 0

    @property
    def vocab_size(self) -> int:
        return self.query_token + 1

    def value_token(self, value: int) -> int:
        return 1 + self.num_keys + value


def make_associative_recall_batch(
    batch_size: int,
    seq_len: int,
    spec: TaskSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = min((seq_len - 2) // 2, spec.num_keys)
    x = torch.full((batch_size, seq_len), spec.pad_token, dtype=torch.long, device=device)
    y = torch.empty((batch_size,), dtype=torch.long, device=device)
    for b in range(batch_size):
        keys = torch.randperm(spec.num_keys, device=device)[:pairs] + 1
        values = torch.randint(0, spec.num_values, (pairs,), device=device)
        query_idx = int(torch.randint(0, pairs, (1,), device=device).item())
        x[b, 0 : 2 * pairs : 2] = keys
        x[b, 1 : 2 * pairs : 2] = torch.tensor(
            [spec.value_token(int(v.item())) for v in values], device=device
        )
        x[b, -2] = spec.query_token
        x[b, -1] = keys[query_idx]
        y[b] = values[query_idx]
    return x, y


def make_copy_batch(
    batch_size: int,
    seq_len: int,
    spec: TaskSpec,
    device: torch.device,
    source_pos: int,
    seed: int | None = None,
    batch_index: int | None = None,
    stream: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is None or batch_index is None:
        x = torch.randint(
            1,
            spec.num_values + 1,
            (batch_size, seq_len),
            dtype=torch.long,
            device=device,
        )
    else:
        derived_seed = stable_int_seed(seed, batch_index, seq_len, spec.num_values, stream)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(derived_seed)
        x = torch.randint(
            1,
            spec.num_values + 1,
            (batch_size, seq_len),
            dtype=torch.long,
            generator=gen,
        ).to(device)
    y = x[:, source_pos] - 1
    return x, y


def stable_int_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def canonical_task_name(task: str) -> str:
    if task == "copy":
        return "copy_first"
    return task


def make_batch(
    args,
    spec: TaskSpec,
    device: torch.device,
    seq_len: int | None = None,
    batch_size: int | None = None,
    batch_index: int | None = None,
    stream: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    task = canonical_task_name(args.task)
    seq_len = int(seq_len if seq_len is not None else args.seq_len)
    batch_size = int(batch_size if batch_size is not None else args.batch_size)
    if args.task == "associative_recall":
        return make_associative_recall_batch(batch_size, seq_len, spec, device)
    if task == "copy_visible":
        return make_copy_batch(
            batch_size,
            seq_len,
            spec,
            device,
            source_pos=seq_len - 2,
            seed=getattr(args, "seed", None),
            batch_index=batch_index,
            stream=stream,
        )
    if task == "copy_first":
        return make_copy_batch(
            batch_size,
            seq_len,
            spec,
            device,
            source_pos=0,
            seed=getattr(args, "seed", None),
            batch_index=batch_index,
            stream=stream,
        )
    raise ValueError(f"unknown task: {args.task}")


RESULT_FIELDS = [
    "run_id",
    "timestamp",
    "host",
    "local_or_remote",
    "git_commit",
    "config_path",
    "config_sha256",
    "command",
    "output_dir",
    "log_path",
    "CUDA_VISIBLE_DEVICES",
    "gpu_name",
    "torch_version",
    "task",
    "data_mode",
    "num_values",
    "method",
    "attention_backend",
    "N_train",
    "N_eval",
    "B",
    "d",
    "G_type",
    "H_type",
    "seed",
    "architecture",
    "layers",
    "d_model",
    "heads",
    "ffn_dim",
    "dropout",
    "optimizer",
    "learning_rate",
    "steps",
    "batch_size",
    "eval_batches",
    "raw_K",
    "effective_K_mean",
    "effective_K_min",
    "effective_K_max",
    "duplicate_rate",
    "self_loop_rate",
    "attention_pair_count",
    "final_train_loss",
    "eval_loss",
    "eval_accuracy",
    "final_valid_loss",
    "final_valid_accuracy",
    "tokens_per_sec",
    "elapsed_sec",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "artifact_dir",
    "metrics_path",
    "neighbor_shape",
    "block_pair_shape",
    "status",
    "failure_reason",
]


def resolve_attention_backend(requested: str, method: str) -> str:
    if requested == "auto":
        return "dense_mask" if method == "dense" else "neighbor"
    if requested == "auto_split":
        return "dense_mask" if method == "dense" else "split"
    if requested == "auto_blockpair":
        return "dense_mask" if method == "dense" else "blockpair"
    if requested in {"neighbor", "split", "blockpair"} and method == "dense":
        raise ValueError(
            "dense method with sparse backend would use K=N; use dense_mask, auto, auto_split, or auto_blockpair"
        )
    return requested


def make_attention_artifacts(
    method: str,
    seq_len: int,
    args,
    device: torch.device,
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict]:
    mask = build_attention_mask(
        method,
        seq_len,
        args.block_size,
        args.degree,
        device,
        args.seed,
        getattr(args, "graph_config", None),
    )
    neighbors = None
    valid_neighbors = None
    block_pair_index = None
    if attention_backend == "neighbor":
        neighbors, valid_neighbors = mask_to_neighbors(mask)
    elif attention_backend in {"split", "blockpair"}:
        cross_mask = build_cross_mask(
            method,
            seq_len,
            args.block_size,
            args.degree,
            device,
            args.seed,
            getattr(args, "graph_config", None),
        )
        neighbors, valid_neighbors = mask_to_neighbors(cross_mask)
        if attention_backend == "blockpair":
            block_pair_index = cross_neighbors_to_block_pair_index(
                neighbors, valid_neighbors, args.block_size
            )
    return mask, neighbors, valid_neighbors, block_pair_index, mask_metrics(
        mask, method, args.block_size, args.degree
    )


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(
    model,
    mask,
    neighbors,
    valid_neighbors,
    block_pair_index,
    args,
    spec,
    device,
    seq_len: int,
    stream: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx in range(args.eval_batches):
            x, y = make_batch(
                args,
                spec,
                device,
                seq_len=seq_len,
                batch_size=args.batch_size,
                batch_index=batch_idx,
                stream=stream,
            )
            logits = model(x, mask, neighbors, valid_neighbors, block_pair_index)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite eval loss for {stream}")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += y.numel()
    model.train()
    return total_loss / args.eval_batches, total_correct / total


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else RESULT_FIELDS
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_method(
    method: str,
    args,
    device: torch.device,
    output_dir: Path,
    train_len: int | None = None,
    seed: int | None = None,
    eval_lengths: list[int] | None = None,
) -> dict:
    train_len = int(train_len if train_len is not None else args.seq_len)
    seed = int(seed if seed is not None else args.seed)
    eval_lengths = [int(v) for v in (eval_lengths if eval_lengths is not None else [train_len])]
    args = copy.copy(args)
    args.seq_len = train_len
    args.seed = seed
    set_seed(args.seed)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    spec = TaskSpec(num_keys=args.num_keys, num_values=args.num_values)
    attention_backend = resolve_attention_backend(args.attention_backend, method)
    mask, neighbors, valid_neighbors, block_pair_index, train_mask_metric = make_attention_artifacts(
        method, train_len, args, device, attention_backend
    )
    model = TinyTransformer(
        vocab_size=spec.vocab_size,
        num_classes=spec.num_values,
        seq_len=max([train_len, *eval_lengths]),
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attention_backend=attention_backend,
        block_size=args.block_size,
    ).to(device)
    if args.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer: {args.optimizer}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        cuda_sync(device)
    start = time.perf_counter()
    last_loss = None
    metrics_path = output_dir / "metrics.jsonl"
    method_metrics_path = output_dir / f"{method}_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fp:
        for step in range(1, args.steps + 1):
            x, y = make_batch(
                args,
                spec,
                device,
                seq_len=train_len,
                batch_size=args.batch_size,
                batch_index=step,
                stream="train",
            )
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask, neighbors, valid_neighbors, block_pair_index)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method} produced non-finite loss at step {step}")
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_loss, eval_acc = evaluate(
                    model,
                    mask,
                    neighbors,
                    valid_neighbors,
                    block_pair_index,
                    args,
                    spec,
                    device,
                    train_len,
                    stream=f"valid_N{train_len}",
                )
                row = {
                    "step": step,
                    "method": method,
                    "seed": seed,
                    "N_train": train_len,
                    "N_eval": train_len,
                    "train_loss": last_loss,
                    "valid_loss": eval_loss,
                    "valid_accuracy": eval_acc,
                }
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                print(json.dumps(row), flush=True)
    if torch.cuda.is_available():
        cuda_sync(device)
    elapsed = time.perf_counter() - start
    method_metrics_path.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_result = {
        "method": method,
        "task": args.task,
        "attention_backend": attention_backend,
        "seq_len": train_len,
        "block_size": args.block_size,
        "degree": args.degree,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "final_train_loss": last_loss,
        "tokens_per_sec": args.steps * args.batch_size * args.seq_len / elapsed,
        "elapsed_sec": elapsed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "mask": train_mask_metric,
        "task_spec": asdict(spec),
        "metrics_path": str(metrics_path),
        "neighbor_shape": list(neighbors.shape) if neighbors is not None else None,
        "block_pair_shape": list(block_pair_index.shape) if block_pair_index is not None else None,
    }
    records = []
    for eval_len in eval_lengths:
        eval_mask, eval_neighbors, eval_valid_neighbors, eval_block_pair_index, eval_mask_metric = (
            make_attention_artifacts(method, eval_len, args, device, attention_backend)
        )
        eval_loss, eval_acc = evaluate(
            model,
            eval_mask,
            eval_neighbors,
            eval_valid_neighbors,
            eval_block_pair_index,
            args,
            spec,
            device,
            eval_len,
            stream=f"eval_N{eval_len}",
        )
        graph_config = getattr(args, "graph_config", DEFAULT_GRAPH_CONFIG)
        g_type = graph_config.get("G", {}).get("type")
        h_type = graph_config.get("H", {}).get("type")
        run_id = f"train_N{train_len}_seed{seed}_{method}"
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "local_or_remote": getattr(args, "local_or_remote", "unknown"),
            "git_commit": getattr(args, "git_commit", ""),
            "config_path": getattr(args, "config_path", ""),
            "config_sha256": getattr(args, "config_sha256", ""),
            "command": getattr(args, "command", ""),
            "output_dir": str(getattr(args, "output_dir", "")),
            "log_path": getattr(args, "log_path", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "task": args.task,
            "data_mode": getattr(args, "data_mode", "online"),
            "num_values": args.num_values,
            "method": method,
            "attention_backend": attention_backend,
            "N_train": train_len,
            "N_eval": eval_len,
            "B": args.block_size,
            "d": args.degree,
            "G_type": g_type,
            "H_type": h_type,
            "seed": seed,
            "architecture": args.architecture,
            "layers": args.layers,
            "d_model": args.d_model,
            "heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "raw_K": eval_mask_metric["raw_k"],
            "effective_K_mean": eval_mask_metric["effective_k_mean"],
            "effective_K_min": eval_mask_metric["effective_k_min"],
            "effective_K_max": eval_mask_metric["effective_k_max"],
            "duplicate_rate": eval_mask_metric["duplicate_rate"],
            "self_loop_rate": eval_mask_metric["self_loop_rate"],
            "attention_pair_count": eval_mask_metric["attention_pair_count"],
            "final_train_loss": last_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
            "final_valid_loss": eval_loss,
            "final_valid_accuracy": eval_acc,
            "tokens_per_sec": args.steps * args.batch_size * train_len / elapsed,
            "elapsed_sec": elapsed,
            "peak_allocated_gb": base_result["peak_allocated_gb"],
            "peak_reserved_gb": base_result["peak_reserved_gb"],
            "artifact_dir": str(output_dir),
            "metrics_path": str(metrics_path),
            "neighbor_shape": base_result["neighbor_shape"],
            "block_pair_shape": base_result["block_pair_shape"],
            "status": "ok",
            "failure_reason": "",
        }
        records.append(record)
    base_result["evals"] = records
    if records:
        base_result["final_valid_loss"] = records[0]["eval_loss"]
        base_result["final_valid_accuracy"] = records[0]["eval_accuracy"]
    write_csv(output_dir / "results.csv", records, RESULT_FIELDS)
    write_jsonl(output_dir / "results.jsonl", records)
    write_json(
        output_dir / "summary.json",
        {
            "status": "ok",
            "run_id": f"train_N{train_len}_seed{seed}_{method}",
            "config": serialize_args(args),
            "result": base_result,
            "results": records,
        },
    )
    return base_result


DEFAULT_CONFIG = {
    "task": {
        "name": "copy",
        "data": "online",
        "num_values": 4,
        "train_lengths": [128],
        "eval_lengths": [128],
    },
    "model": {
        "architecture": "tiny_transformer",
        "layers": 2,
        "d_model": 64,
        "heads": 4,
        "ffn_dim": 128,
        "dropout": 0.1,
        "attention_backend": "auto_split",
    },
    "attention": {
        "methods": ["dense", "local", "random", "zigzag"],
        "block_size": 16,
        "degree": 2,
        "graph": copy.deepcopy(DEFAULT_GRAPH_CONFIG),
    },
    "train": {
        "steps": 10,
        "batch_size": 4,
        "eval_batches": 2,
        "learning_rate": 1e-3,
        "seeds": [0],
        "optimizer": "adamw",
        "log_every": 5,
    },
    "output": {
        "root": "outputs/synthetic_mvp",
    },
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def file_sha256(path: Path | None) -> str:
    if path is None:
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    env_commit = os.environ.get("COPY_V05_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def detect_location() -> str:
    cwd = str(Path.cwd())
    if cwd.startswith("/home/huiwei/ysx/zigzag_attention"):
        return "remote"
    if cwd.startswith("/Users/sxye/Documents/expander"):
        return "local"
    return "unknown"


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def serialize_args(args) -> dict:
    return jsonable(vars(args))


def shell_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def write_command_script(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    commit = os.environ.get("COPY_V05_GIT_COMMIT") or os.environ.get("GIT_COMMIT", "")
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(Path.cwd()))}",
                "conda activate ysx_base 2>/dev/null || true",
                (
                    f"COPY_V05_GIT_COMMIT={shlex.quote(commit)} "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda)} {command}"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def load_config(path: Path | None) -> tuple[dict, str, str]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG), "", ""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, loaded), str(path), file_sha256(path)


def apply_cli_overrides(config: dict, cli) -> dict:
    config = copy.deepcopy(config)
    if cli.task is not None:
        config["task"]["name"] = "copy" if cli.task == "copy_first" else cli.task
    if cli.seq_len is not None:
        config["task"]["train_lengths"] = [cli.seq_len]
        config["task"]["eval_lengths"] = [cli.seq_len]
    if cli.train_lengths is not None:
        config["task"]["train_lengths"] = parse_csv_ints(cli.train_lengths)
    if cli.eval_lengths is not None:
        config["task"]["eval_lengths"] = parse_csv_ints(cli.eval_lengths)
    if cli.methods is not None:
        config["attention"]["methods"] = parse_csv_strings(cli.methods)
    if cli.block_size is not None:
        config["attention"]["block_size"] = cli.block_size
    if cli.degree is not None:
        config["attention"]["degree"] = cli.degree
    if cli.steps is not None:
        config["train"]["steps"] = cli.steps
    if cli.eval_batches is not None:
        config["train"]["eval_batches"] = cli.eval_batches
    if cli.batch_size is not None:
        config["train"]["batch_size"] = cli.batch_size
    if cli.learning_rate is not None:
        config["train"]["learning_rate"] = cli.learning_rate
    if cli.seed is not None:
        config["train"]["seeds"] = [cli.seed]
    if cli.seeds is not None:
        config["train"]["seeds"] = parse_csv_ints(cli.seeds)
    if cli.log_every is not None:
        config["train"]["log_every"] = cli.log_every
    if cli.d_model is not None:
        config["model"]["d_model"] = cli.d_model
    if cli.layers is not None:
        config["model"]["layers"] = cli.layers
    if cli.heads is not None:
        config["model"]["heads"] = cli.heads
    if cli.ffn_dim is not None:
        config["model"]["ffn_dim"] = cli.ffn_dim
    if cli.dropout is not None:
        config["model"]["dropout"] = cli.dropout
    if cli.attention_backend is not None:
        config["model"]["attention_backend"] = cli.attention_backend
    if cli.num_values is not None:
        config["task"]["num_values"] = cli.num_values
    if cli.num_keys is not None:
        config["task"]["num_keys"] = cli.num_keys
    if cli.output_dir is not None:
        config["output"]["root"] = str(cli.output_dir)
    return config


def build_runtime_args(config: dict, cli, config_path: str, config_sha: str) -> SimpleNamespace:
    task = config["task"]
    model = config["model"]
    attention = config["attention"]
    train = config["train"]
    output = config["output"]
    train_lengths = [int(v) for v in task.get("train_lengths", task.get("sequence_lengths", [128]))]
    eval_lengths = [int(v) for v in task.get("eval_lengths", train_lengths)]
    if model.get("architecture") != "tiny_transformer":
        raise ValueError(f"unsupported architecture: {model.get('architecture')}")
    graph_config = validate_graph_config(
        max([*train_lengths, *eval_lengths]),
        int(attention["block_size"]),
        int(attention["degree"]),
        attention.get("graph", DEFAULT_GRAPH_CONFIG),
    )
    for seq_len in [*train_lengths, *eval_lengths]:
        validate_graph_config(
            seq_len,
            int(attention["block_size"]),
            int(attention["degree"]),
            graph_config,
        )
    steps = int(train["steps"])
    log_every = int(train.get("log_every", max(1, steps // 10)))
    command = shell_command()
    return SimpleNamespace(
        task=canonical_task_name(task.get("name", "copy")),
        data_mode=task.get("data", "online"),
        num_values=int(task.get("num_values", 4)),
        num_keys=int(task.get("num_keys", 64)),
        train_lengths=train_lengths,
        eval_lengths=eval_lengths,
        methods=[str(method) for method in attention["methods"]],
        block_size=int(attention["block_size"]),
        degree=int(attention["degree"]),
        graph_config=graph_config,
        architecture=model.get("architecture", "tiny_transformer"),
        layers=int(model["layers"]),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        ffn_dim=int(model["ffn_dim"]),
        dropout=float(model["dropout"]),
        attention_backend=model["attention_backend"],
        steps=steps,
        batch_size=int(train["batch_size"]),
        eval_batches=int(train["eval_batches"]),
        learning_rate=float(train["learning_rate"]),
        seeds=[int(seed) for seed in train["seeds"]],
        optimizer=train.get("optimizer", "adamw").lower(),
        log_every=log_every,
        output_dir=Path(output["root"]),
        device=cli.device,
        skip_tests=bool(cli.skip_tests),
        config_path=config_path,
        config_sha256=config_sha,
        config_snapshot=config,
        command=command,
        log_path=cli.log_path or os.environ.get("COPY_V05_LOG_PATH", ""),
        git_commit=git_commit(),
        local_or_remote=cli.local_or_remote or detect_location(),
        seq_len=train_lengths[0],
        seed=int(train["seeds"][0]),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=["copy", "copy_first", "copy_visible", "associative_recall"])
    parser.add_argument("--methods")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--train-lengths")
    parser.add_argument("--eval-lengths")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--ffn-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--attention-backend",
        choices=["dense_mask", "neighbor", "split", "blockpair", "auto", "auto_split", "auto_blockpair"],
        help=(
            "dense_mask keeps the debug N x N score path; auto uses neighbor tables "
            "for sparse methods; auto_split uses local/cross split for sparse methods; "
            "auto_blockpair groups cross edges by block pair."
        ),
    )
    parser.add_argument("--num-keys", type=int)
    parser.add_argument("--num-values", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-or-remote", choices=["local", "remote", "unknown"])
    parser.add_argument("--log-path")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def failure_record(args, run_id: str, train_len: int, seed: int, method: str, error: Exception, run_dir: Path) -> dict:
    graph_config = getattr(args, "graph_config", DEFAULT_GRAPH_CONFIG)
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "local_or_remote": args.local_or_remote,
        "git_commit": args.git_commit,
        "config_path": args.config_path,
        "config_sha256": args.config_sha256,
        "command": args.command,
        "output_dir": str(args.output_dir),
        "log_path": args.log_path,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "task": args.task,
        "data_mode": args.data_mode,
        "num_values": args.num_values,
        "method": method,
        "attention_backend": args.attention_backend,
        "N_train": train_len,
        "N_eval": "",
        "B": args.block_size,
        "d": args.degree,
        "G_type": graph_config.get("G", {}).get("type"),
        "H_type": graph_config.get("H", {}).get("type"),
        "seed": seed,
        "architecture": args.architecture,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "raw_K": "",
        "effective_K_mean": "",
        "effective_K_min": "",
        "effective_K_max": "",
        "duplicate_rate": "",
        "self_loop_rate": "",
        "attention_pair_count": "",
        "final_train_loss": "",
        "eval_loss": "",
        "eval_accuracy": "",
        "final_valid_loss": "",
        "final_valid_accuracy": "",
        "tokens_per_sec": "",
        "elapsed_sec": "",
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
        "artifact_dir": str(run_dir),
        "metrics_path": "",
        "neighbor_shape": "",
        "block_pair_shape": "",
        "status": "failed",
        "failure_reason": repr(error),
    }


def main() -> None:
    cli = parse_args()
    config, config_path, config_sha = load_config(cli.config)
    config = apply_cli_overrides(config, cli)
    args = build_runtime_args(config, cli, config_path, config_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    set_seed(args.seed)
    write_json(args.output_dir / "config_snapshot.json", args.config_snapshot)
    write_command_script(args.output_dir / "command.sh", args.command)

    test_results = []
    if not args.skip_tests:
        test_results = run_mask_tests(device)
        write_json(args.output_dir / "mask_tests.json", test_results)
        print(json.dumps({"mask_tests": "ok", "cases": len(test_results)}), flush=True)

    all_records: list[dict] = []
    method_results: list[dict] = []
    metrics_lines: list[str] = []
    for train_len in args.train_lengths:
        for seed in args.seeds:
            for method in args.methods:
                run_id = f"train_N{train_len}_seed{seed}_{method}"
                run_dir = args.output_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                run_args = copy.copy(args)
                run_args.output_dir = run_dir
                write_json(run_dir / "config_snapshot.json", args.config_snapshot)
                write_command_script(run_dir / "command.sh", args.command)
                try:
                    result = train_method(
                        method,
                        run_args,
                        device,
                        run_dir,
                        train_len=train_len,
                        seed=seed,
                        eval_lengths=args.eval_lengths,
                    )
                    method_results.append(result)
                    all_records.extend(result["evals"])
                    metrics_lines.extend((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
                    print(json.dumps({"completed": run_id, "evals": result["evals"]}, indent=2), flush=True)
                except Exception as exc:
                    (run_dir / "error.log").write_text(repr(exc) + "\n", encoding="utf-8")
                    failed = failure_record(args, run_id, train_len, seed, method, exc, run_dir)
                    all_records.append(failed)
                    write_csv(run_dir / "results.csv", [failed], RESULT_FIELDS)
                    write_jsonl(run_dir / "results.jsonl", [failed])
                    write_json(
                        run_dir / "summary.json",
                        {
                            "status": "failed",
                            "run_id": run_id,
                            "config": serialize_args(args),
                            "failure_reason": repr(exc),
                            "results": [failed],
                        },
                    )
                    print(json.dumps({"failed": run_id, "error": repr(exc)}), flush=True)

    if metrics_lines:
        (args.output_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "results.csv", all_records, RESULT_FIELDS)
    write_jsonl(args.output_dir / "results.jsonl", all_records)
    write_csv(args.output_dir / "phase3_results.csv", all_records, RESULT_FIELDS)
    write_jsonl(args.output_dir / "phase3_results.jsonl", all_records)
    status = "ok" if all(row.get("status") == "ok" for row in all_records) else "failed"
    write_json(
        args.output_dir / "summary.json",
        {
            "status": status,
            "config": serialize_args(args),
            "mask_test_cases": len(test_results),
            "results": all_records,
            "method_results": method_results,
        },
    )
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
