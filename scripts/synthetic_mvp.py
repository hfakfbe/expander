import argparse
import csv
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def h_neighbors(port: int, block_size: int, degree: int) -> list[int]:
    if degree >= block_size:
        raise ValueError("degree must be smaller than block_size")
    offsets: list[int] = []
    step = 1
    while len(offsets) < degree:
        offsets.append(step)
        if len(offsets) < degree:
            offsets.append(-step)
        step += 1
    return [int((port + off) % block_size) for off in offsets[:degree]]


def rot_g_cyclic(block: int, port: int, num_blocks: int, block_size: int) -> tuple[int, int]:
    max_offset = max(1, num_blocks // 2)
    offset = (port // 2) % max_offset + 1
    if port % 2 == 0:
        return (block + offset) % num_blocks, port ^ 1
    return (block - offset) % num_blocks, port ^ 1


def build_local_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return (idx[:, None] // block_size) == (idx[None, :] // block_size)


def build_zigzag_cross(seq_len: int, block_size: int, degree: int, device: torch.device) -> torch.Tensor:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    num_blocks = seq_len // block_size
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for v in range(num_blocks):
        for i in range(block_size):
            src = v * block_size + i
            for i_prime in h_neighbors(i, block_size, degree):
                w, j_prime = rot_g_cyclic(v, i_prime, num_blocks, block_size)
                for j in h_neighbors(j_prime, block_size, degree):
                    dst = w * block_size + j
                    mask[src, dst] = True
    return mask


def build_random_cross(
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    k = degree * degree
    all_idx = torch.arange(seq_len)
    for src in range(seq_len):
        block = src // block_size
        nonlocal_idx = all_idx[all_idx // block_size != block]
        perm = torch.randperm(len(nonlocal_idx), generator=gen)[:k]
        mask[src, nonlocal_idx[perm].to(device)] = True
    return mask


def build_attention_mask(
    method: str,
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if method == "dense":
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    local = build_local_mask(seq_len, block_size, device)
    if method == "local":
        return local
    if method == "random":
        return local | build_random_cross(seq_len, block_size, degree, device, seed)
    if method == "zigzag":
        return local | build_zigzag_cross(seq_len, block_size, degree, device)
    raise ValueError(f"unknown method: {method}")


def build_cross_mask(
    method: str,
    seq_len: int,
    block_size: int,
    degree: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    local = build_local_mask(seq_len, block_size, device)
    if method in {"dense", "local"}:
        return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    if method == "random":
        return build_random_cross(seq_len, block_size, degree, device, seed) & ~local
    if method == "zigzag":
        return build_zigzag_cross(seq_len, block_size, degree, device) & ~local
    raise ValueError(f"unknown method: {method}")


def expected_raw_k(method: str, seq_len: int, block_size: int, degree: int) -> int:
    if method == "dense":
        return seq_len
    if method == "local":
        return block_size
    return block_size + degree * degree


def mask_metrics(mask: torch.Tensor, method: str, block_size: int, degree: int) -> dict:
    seq_len = mask.shape[0]
    raw_k = expected_raw_k(method, seq_len, block_size, degree)
    effective = mask.sum(dim=-1).float()
    duplicate_rate = max(0.0, (raw_k - float(effective.mean().item())) / raw_k)
    return {
        "raw_k": raw_k,
        "effective_k_mean": float(effective.mean().item()),
        "effective_k_min": int(effective.min().item()),
        "effective_k_max": int(effective.max().item()),
        "duplicate_rate_estimate": duplicate_rate,
        "attention_pair_count": int(mask.sum().item()),
        "self_loop_rate": float(torch.diag(mask).float().mean().item()),
    }


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
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(1, spec.num_values + 1, (batch_size, seq_len), dtype=torch.long, device=device)
    y = x[:, source_pos] - 1
    return x, y


def make_batch(args, spec: TaskSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if args.task == "associative_recall":
        return make_associative_recall_batch(args.batch_size, args.seq_len, spec, device)
    if args.task == "copy_visible":
        return make_copy_batch(args.batch_size, args.seq_len, spec, device, source_pos=args.seq_len - 2)
    if args.task == "copy_first":
        return make_copy_batch(args.batch_size, args.seq_len, spec, device, source_pos=0)
    raise ValueError(f"unknown task: {args.task}")


def evaluate(model, mask, neighbors, valid_neighbors, block_pair_index, args, spec, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = make_batch(args, spec, device)
            logits = model(x, mask, neighbors, valid_neighbors, block_pair_index)
            total_loss += float(F.cross_entropy(logits, y).item())
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += y.numel()
    model.train()
    return total_loss / args.eval_batches, total_correct / total


def train_method(method: str, args, device: torch.device, output_dir: Path) -> dict:
    set_seed(args.seed)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    spec = TaskSpec(num_keys=args.num_keys, num_values=args.num_values)
    mask = build_attention_mask(method, args.seq_len, args.block_size, args.degree, device, args.seed)
    attention_backend = args.attention_backend
    if attention_backend == "auto":
        attention_backend = "dense_mask" if method == "dense" else "neighbor"
    if attention_backend == "auto_split":
        attention_backend = "dense_mask" if method == "dense" else "split"
    if attention_backend == "auto_blockpair":
        attention_backend = "dense_mask" if method == "dense" else "blockpair"
    if attention_backend in {"neighbor", "split", "blockpair"} and method == "dense":
        raise ValueError("dense method with sparse backend would use K=N; use dense_mask, auto, auto_split, or auto_blockpair")
    neighbors = None
    valid_neighbors = None
    block_pair_index = None
    if attention_backend == "neighbor":
        neighbors, valid_neighbors = mask_to_neighbors(mask)
    elif attention_backend in {"split", "blockpair"}:
        cross_mask = build_cross_mask(
            method, args.seq_len, args.block_size, args.degree, device, args.seed
        )
        neighbors, valid_neighbors = mask_to_neighbors(cross_mask)
        if attention_backend == "blockpair":
            block_pair_index = cross_neighbors_to_block_pair_index(
                neighbors, valid_neighbors, args.block_size
            )
    model = TinyTransformer(
        vocab_size=spec.vocab_size,
        num_classes=spec.num_values,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attention_backend=attention_backend,
        block_size=args.block_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = None
    metrics_path = output_dir / f"{method}_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fp:
        for step in range(1, args.steps + 1):
            x, y = make_batch(args, spec, device)
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
                    model, mask, neighbors, valid_neighbors, block_pair_index, args, spec, device
                )
                row = {
                    "step": step,
                    "method": method,
                    "train_loss": last_loss,
                    "valid_loss": eval_loss,
                    "valid_accuracy": eval_acc,
                }
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                print(json.dumps(row), flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    eval_loss, eval_acc = evaluate(
        model, mask, neighbors, valid_neighbors, block_pair_index, args, spec, device
    )
    result = {
        "method": method,
        "task": args.task,
        "attention_backend": attention_backend,
        "seq_len": args.seq_len,
        "block_size": args.block_size,
        "degree": args.degree,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "final_train_loss": last_loss,
        "final_valid_loss": eval_loss,
        "final_valid_accuracy": eval_acc,
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
        "mask": mask_metrics(mask, method, args.block_size, args.degree),
        "task_spec": asdict(spec),
        "metrics_path": str(metrics_path),
        "neighbor_shape": list(neighbors.shape) if neighbors is not None else None,
        "block_pair_shape": list(block_pair_index.shape) if block_pair_index is not None else None,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="associative_recall",
        choices=["associative_recall", "copy_visible", "copy_first"],
    )
    parser.add_argument("--methods", default="dense,local,random,zigzag")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--attention-backend",
        default="dense_mask",
        choices=["dense_mask", "neighbor", "split", "blockpair", "auto", "auto_split", "auto_blockpair"],
        help=(
            "dense_mask keeps the debug N x N score path; auto uses neighbor tables "
            "for sparse methods; auto_split uses local/cross split for sparse methods; "
            "auto_blockpair groups cross edges by block pair."
        ),
    )
    parser.add_argument("--num-keys", type=int, default=64)
    parser.add_argument("--num-values", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_mvp"))
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    test_results = []
    if not args.skip_tests:
        test_results = run_mask_tests(device)
        (args.output_dir / "mask_tests.json").write_text(
            json.dumps(test_results, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"mask_tests": "ok", "cases": len(test_results)}), flush=True)

    all_results = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        result = train_method(method, args, device, args.output_dir)
        all_results.append(result)
        print(json.dumps({"completed": method, "result": result}, indent=2), flush=True)

    summary = {
        "status": "ok",
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "mask_test_cases": len(test_results),
        "results": all_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_path = args.output_dir / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "task",
            "method",
            "attention_backend",
            "N",
            "B",
            "d",
            "raw_K",
            "effective_K_mean",
            "attention_pair_count",
            "final_valid_loss",
            "final_valid_accuracy",
            "tokens_per_sec",
            "peak_allocated_gb",
            "peak_reserved_gb",
            "device",
            "neighbor_shape",
            "block_pair_shape",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(
                {
                    "task": result["task"],
                    "method": result["method"],
                    "attention_backend": result["attention_backend"],
                    "N": result["seq_len"],
                    "B": result["block_size"],
                    "d": result["degree"],
                    "raw_K": result["mask"]["raw_k"],
                    "effective_K_mean": result["mask"]["effective_k_mean"],
                    "attention_pair_count": result["mask"]["attention_pair_count"],
                    "final_valid_loss": result["final_valid_loss"],
                    "final_valid_accuracy": result["final_valid_accuracy"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "peak_allocated_gb": result["peak_allocated_gb"],
                    "peak_reserved_gb": result["peak_reserved_gb"],
                    "device": result["device"],
                    "neighbor_shape": result["neighbor_shape"],
                    "block_pair_shape": result["block_pair_shape"],
                }
            )


if __name__ == "__main__":
    main()
