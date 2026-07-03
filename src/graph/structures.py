from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LayerGraph:
    layer_index: int
    method: str
    seq_len: int
    mask: torch.Tensor
    counts: list[Counter[int]] | None
    log_m: torch.Tensor | None
    seed: int | None


def apply_causal(mask: torch.Tensor, causal: bool) -> torch.Tensor:
    if not causal:
        return mask
    seq_len = int(mask.shape[0])
    return mask & torch.ones((seq_len, seq_len), dtype=torch.bool, device=mask.device).tril()


def dense_mask(seq_len: int, device: torch.device, causal: bool) -> torch.Tensor:
    return apply_causal(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), causal)


def sliding_window_mask(seq_len: int, window_size: int, device: torch.device, causal: bool) -> torch.Tensor:
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    idx = torch.arange(seq_len, device=device)
    distance = (idx[:, None] - idx[None, :]).abs()
    return apply_causal(distance <= int(window_size), causal)


def local_edge_pairs(seq_len: int, window_size: int, causal: bool) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for src in range(seq_len):
        lo = max(0, src - int(window_size))
        hi = min(seq_len, src + int(window_size) + 1)
        for dst in range(lo, hi):
            if not causal or dst <= src:
                out.add((src, dst))
    return out


def _remote_budget(seq_len: int, config: dict, method: str) -> int:
    method_cfg = dict(config.get(method, {}))
    degree = method_cfg.get("degree")
    if degree is None:
        degree = config.get("degree")
    if degree is not None:
        return max(1, int(degree))
    density = method_cfg.get("density", config.get("density"))
    if density is None:
        raise ValueError(f"{method} requires degree or density")
    return max(1, int(round(float(density) * seq_len)))


def random_regular_counts(
    seq_len: int,
    degree: int,
    seed: int,
    *,
    exclude_edges: Iterable[tuple[int, int]] = (),
) -> list[Counter[int]]:
    excluded = set(exclude_edges)
    rows = [Counter() for _ in range(seq_len)]
    rng = random.Random(f"random_regular|{seed}|{seq_len}|{degree}")
    for src in range(seq_len):
        candidates = [dst for dst in range(seq_len) if (src, dst) not in excluded]
        if degree > len(candidates):
            raise ValueError(f"random_regular degree {degree} exceeds candidates for row {src}")
        for dst in rng.sample(candidates, degree):
            rows[src][dst] += 1
    return rows


def _h_neighbors(port: int, block_size: int, degree: int) -> list[int]:
    offsets: list[int] = []
    step = 1
    while len(offsets) < degree:
        offsets.append(step)
        if len(offsets) < degree:
            offsets.append(-step)
        step += 1
    return [int((port + off) % block_size) for off in offsets[:degree]]


def _rot_g(block: int, port: int, num_blocks: int) -> tuple[int, int]:
    max_offset = max(1, num_blocks // 2)
    offset = (port // 2) % max_offset + 1
    if port % 2 == 0:
        return (block + offset) % num_blocks, port ^ 1
    return (block - offset) % num_blocks, port ^ 1


def zigzag_counts(seq_len: int, block_size: int, degree: int) -> list[Counter[int]]:
    if seq_len % block_size != 0:
        raise ValueError("zigzag requires sequence length divisible by B")
    if degree <= 0 or degree >= block_size:
        raise ValueError("zigzag requires 0 < d < B")
    num_blocks = seq_len // block_size
    rows = [Counter() for _ in range(seq_len)]
    for block in range(num_blocks):
        for port in range(block_size):
            src = block * block_size + port
            for mid_port in _h_neighbors(port, block_size, degree):
                dst_block, dst_port = _rot_g(block, mid_port, num_blocks)
                for final_port in _h_neighbors(dst_port, block_size, degree):
                    rows[src][dst_block * block_size + final_port] += 1
    return rows


def counts_to_mask(rows: list[Counter[int]], seq_len: int, device: torch.device, causal: bool) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for src, counts in enumerate(rows):
        if not counts:
            continue
        dst = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
        mask[src, dst] = True
    return apply_causal(mask, causal)


def counts_to_log_m(rows: list[Counter[int]], seq_len: int, device: torch.device, causal: bool) -> torch.Tensor:
    log_m = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    for src, counts in enumerate(rows):
        for dst, multiplicity in counts.items():
            if not causal or int(dst) <= src:
                log_m[src, int(dst)] = math.log(float(multiplicity))
    return log_m


def merge_counts(base: list[Counter[int]], extra: list[Counter[int]], *, boolean: bool = False) -> list[Counter[int]]:
    rows = [Counter(counts) for counts in base]
    for src, counts in enumerate(extra):
        for dst, value in counts.items():
            if boolean:
                rows[src][int(dst)] = 1
            else:
                rows[src][int(dst)] += int(value)
    if boolean:
        for counts in rows:
            for dst in list(counts):
                counts[dst] = 1
    return rows


def counts_from_edges(seq_len: int, edges: Iterable[tuple[int, int]]) -> list[Counter[int]]:
    rows = [Counter() for _ in range(seq_len)]
    for src, dst in edges:
        rows[int(src)][int(dst)] += 1
    return rows
