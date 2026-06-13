from __future__ import annotations

import torch
import torch.nn as nn


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

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()

def local_valid_from_mask(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    seq_len = mask.shape[0]
    offsets = torch.arange(block_size, device=mask.device)
    block_starts = (torch.arange(seq_len, device=mask.device) // block_size) * block_size
    local_positions = block_starts[:, None] + offsets[None, :]
    return mask.gather(1, local_positions)

def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    log_m: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if log_m is not None:
        scores = scores + log_m[None, None, :, :]
    scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1), v)

def neighbor_attention_from_table(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,
    valid: torch.Tensor,
    log_m: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    gathered_k = k[:, :, neighbors.reshape(-1), :].reshape(
        batch, heads, seq_len, neighbors.shape[1], head_dim
    )
    gathered_v = v[:, :, neighbors.reshape(-1), :].reshape(
        batch, heads, seq_len, neighbors.shape[1], head_dim
    )
    scores = (q[:, :, :, None, :] * gathered_k).sum(dim=-1) * (head_dim ** -0.5)
    if log_m is not None:
        scores = scores + log_m[None, None, :, :]
    scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return (weights[..., None] * gathered_v).sum(dim=-2)

def neighbor_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    log_m: torch.Tensor | None = None,
) -> torch.Tensor:
    neighbors, valid = mask_to_neighbors(mask)
    gathered_log_m = None
    if log_m is not None:
        gathered_log_m = log_m.gather(1, neighbors).masked_fill(~valid, 0.0)
    return neighbor_attention_from_table(q, k, v, neighbors, valid, gathered_log_m)

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
    local_valid: torch.Tensor,
    cross_neighbors: torch.Tensor | None,
    valid_cross_neighbors: torch.Tensor | None,
    local_log_m: torch.Tensor | None = None,
    cross_log_m: torch.Tensor | None = None,
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
    if local_log_m is not None:
        local_scores_flat = local_scores_flat + local_log_m[None, None, :, :]
    local_scores_flat = local_scores_flat.masked_fill(
        ~local_valid[None, None, :, :],
        torch.finfo(local_scores_flat.dtype).min,
    )

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
        if cross_log_m is not None:
            cross_scores = cross_scores + cross_log_m[None, None, :, :]
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
    local_valid: torch.Tensor,
    cross_neighbors: torch.Tensor | None,
    valid_cross_neighbors: torch.Tensor | None,
    block_pair_index: torch.Tensor | None,
    local_log_m: torch.Tensor | None = None,
    cross_log_m: torch.Tensor | None = None,
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
    if local_log_m is not None:
        local_scores_flat = local_scores_flat + local_log_m[None, None, :, :]
    local_scores_flat = local_scores_flat.masked_fill(
        ~local_valid[None, None, :, :],
        torch.finfo(local_scores_flat.dtype).min,
    )

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
        if cross_log_m is not None:
            edge_scores = edge_scores + cross_log_m[src_tokens, slots][None, None, :]
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
