from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from graph_structures import (
    build_graph_artifact,
    build_local_mask,
    build_zigzag_multiplicity,
    counts_to_mask,
    mask_metrics,
)

from .artifacts import counts_to_log_m_matrix, local_log_m_from_matrix
from .attention import (
    cross_neighbors_to_block_pair_index,
    dense_attention,
    local_blockpair_attention,
    local_cross_attention,
    local_valid_from_mask,
    mask_to_neighbors,
    neighbor_attention,
    neighbor_attention_from_table,
)


def _causal_filter_counts(rows: list[Counter[int]]) -> list[Counter[int]]:
    filtered: list[Counter[int]] = []
    for src, counts in enumerate(rows):
        filtered.append(Counter({dst: count for dst, count in counts.items() if dst <= src}))
    return filtered

def _repeated_table_from_counts(
    rows: list[Counter[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(sum(counts.values()) for counts in rows)
    neighbors = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    for src, counts in enumerate(rows):
        slot = 0
        for dst, count in sorted(counts.items()):
            for _ in range(int(count)):
                neighbors[src, slot] = int(dst)
                valid[src, slot] = True
                slot += 1
    return neighbors, valid

def run_mask_tests(device: torch.device) -> list[dict]:
    results = []
    for seq_len in (64, 128):
        for block_size in (16, 32):
            if seq_len % block_size != 0:
                continue
            degree = 4
            if degree >= block_size:
                continue
            artifact = build_graph_artifact(
                N_task=seq_len // 2,
                T_raw=seq_len,
                block_size=block_size,
                degree=degree,
                graph_seed=0,
                g_config={"require_derangement": True, "max_parallel_edges_per_block_pair": None},
                h_config={"allow_self_port": False},
            )
            rows_pre = build_zigzag_multiplicity(
                seq_len, block_size, degree, artifact, include_local=True
            )
            rows = _causal_filter_counts(rows_pre)
            mask = counts_to_mask(rows, seq_len, device)
            log_m_matrix = counts_to_log_m_matrix(rows, seq_len, device).detach()
            q = torch.randn(2, 2, seq_len, 16, device=device)
            k = torch.randn(2, 2, seq_len, 16, device=device)
            v = torch.randn(2, 2, seq_len, 16, device=device)

            repeated_neighbors, repeated_valid = _repeated_table_from_counts(rows, device)
            repeated = neighbor_attention_from_table(
                q, k, v, repeated_neighbors, repeated_valid
            )
            unique_neighbors, unique_valid = mask_to_neighbors(mask)
            unique_log_m = log_m_matrix.gather(1, unique_neighbors).masked_fill(~unique_valid, 0.0)
            unique = neighbor_attention_from_table(
                q, k, v, unique_neighbors, unique_valid, unique_log_m
            )
            dense_ref = dense_attention(q, k, v, mask, log_m_matrix)
            boolean = neighbor_attention(q, k, v, mask)

            local_valid = local_valid_from_mask(mask, block_size)
            local_log_m = local_log_m_from_matrix(log_m_matrix, block_size)
            cross_mask = mask & ~build_local_mask(seq_len, block_size, device)
            cross_neighbors, valid_cross_neighbors = mask_to_neighbors(cross_mask)
            cross_log_m = log_m_matrix.gather(1, cross_neighbors).masked_fill(
                ~valid_cross_neighbors, 0.0
            )
            split = local_cross_attention(
                q,
                k,
                v,
                block_size,
                local_valid,
                cross_neighbors,
                valid_cross_neighbors,
                local_log_m=local_log_m,
                cross_log_m=cross_log_m,
            )
            block_pair_index = cross_neighbors_to_block_pair_index(
                cross_neighbors, valid_cross_neighbors, block_size
            )
            blockpair = local_blockpair_attention(
                q,
                k,
                v,
                block_size,
                local_valid,
                cross_neighbors,
                valid_cross_neighbors,
                block_pair_index,
                local_log_m=local_log_m,
                cross_log_m=cross_log_m,
            )
            repeated_unique_error = float((repeated - unique).abs().max().item())
            dense_unique_error = float((dense_ref - unique).abs().max().item())
            split_unique_error = float((split - unique).abs().max().item())
            blockpair_unique_error = float((blockpair - unique).abs().max().item())
            boolean_difference = float((boolean - unique).abs().max().item())
            assert repeated_unique_error < 1e-5, repeated_unique_error
            assert dense_unique_error < 1e-5, dense_unique_error
            assert split_unique_error < 1e-5, split_unique_error
            assert blockpair_unique_error < 1e-5, blockpair_unique_error
            assert boolean_difference > 0.0, "boolean and unique+logM unexpectedly identical"

            q_grad = q.detach().clone().requires_grad_(True)
            log_m_no_grad = unique_log_m.detach()
            out = neighbor_attention_from_table(
                q_grad, k.detach(), v.detach(), unique_neighbors, unique_valid, log_m_no_grad
            )
            out.sum().backward()
            assert not log_m_no_grad.requires_grad
            assert q_grad.grad is not None

            pre_k = [len(counts) for counts in rows_pre]
            post_k = [len(counts) for counts in rows]
            multiplicities = [count for counts in rows for count in counts.values()]
            row_metric = mask_metrics(mask, "zigzag_certified", block_size, degree)
            row_metric.update(
                {
                    "seq_len": seq_len,
                    "block_size": block_size,
                    "degree": degree,
                    "batch_size": 2,
                    "heads": 2,
                    "d_model": 32,
                    "graph_id": artifact["graph_id"],
                    "pre_causal_unique_K_mean": float(np.mean(pre_k)),
                    "post_causal_unique_K_mean": float(np.mean(post_k)),
                    "pre_causal_pair_count": int(sum(pre_k)),
                    "post_causal_pair_count": int(mask.sum().item()),
                    "multiplicity_max": int(max(multiplicities)),
                    "repeated_unique_logm_max_error": repeated_unique_error,
                    "dense_unique_logm_max_error": dense_unique_error,
                    "split_unique_logm_max_error": split_unique_error,
                    "blockpair_unique_logm_max_error": blockpair_unique_error,
                    "boolean_unique_logm_max_difference": boolean_difference,
                    "logM_requires_grad": bool(log_m_no_grad.requires_grad),
                }
            )
            results.append(row_metric)
    return results
