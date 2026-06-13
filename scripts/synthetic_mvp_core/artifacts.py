from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from graph_structures import (
    DEFAULT_GRAPH_CONFIG,
    build_local_mask,
    build_random_regular_cross_edges,
    build_zigzag_multiplicity,
    canonical_method,
    counts_to_mask,
    mask_metrics,
)

from .attention import (
    build_causal_mask,
    cross_neighbors_to_block_pair_index,
    local_valid_from_mask,
    mask_to_neighbors,
)


RESULT_FIELDS = [
    "version",
    "run_id",
    "status",
    "failure_reason",
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
    "copy_mode",
    "sep_token",
    "eos_token",
    "pad_token",
    "method",
    "graph_id",
    "graph_seed",
    "attention_backend",
    "N_train",
    "N_eval",
    "T_raw",
    "T",
    "B",
    "d",
    "G_type",
    "H_type",
    "causal",
    "multiplicity_mode",
    "seed",
    "architecture",
    "layers",
    "d_model",
    "heads",
    "ffn_dim",
    "dropout",
    "optimizer",
    "learning_rate",
    "log_every",
    "eval_every",
    "steps",
    "batch_size",
    "eval_batches",
    "raw_K",
    "unique_K_mean",
    "effective_K_mean_after_causal",
    "effective_K_min_after_causal",
    "effective_K_max_after_causal",
    "pre_causal_unique_K_mean",
    "pre_causal_pair_count",
    "duplicate_rate",
    "self_loop_rate",
    "attention_pair_count_after_causal",
    "lambda_G",
    "mu_H",
    "rho_bound",
    "rho_exact",
    "certified",
    "graph_certified",
    "implementation_certified",
    "theory_aligned_method",
    "remote_local_overlap_mean",
    "target_in_1hop_rate",
    "target_in_2hop_rate",
    "target_in_Lhop_rate",
    "average_shortest_path",
    "unreachable_rate",
    "final_train_loss",
    "eval_loss",
    "eval_token_accuracy",
    "eval_sequence_accuracy",
    "eval_eos_accuracy",
    "training_curves_path",
    "tokens_per_sec",
    "elapsed_sec",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "artifact_dir",
    "metrics_path",
    "neighbor_shape",
    "block_pair_shape",
]

def method_certification_fields(method: str, certificate: dict, multiplicity_mode: str) -> dict:
    method = canonical_method(method)
    graph_methods = {"zigzag_certified", "zigzag_boolean"}
    graph_certified = bool(certificate.get("certified")) if method in graph_methods else ""
    implementation_certified = ""
    if method == "zigzag_certified":
        implementation_certified = bool(graph_certified and multiplicity_mode == "unique_log_m")
    elif method in {"random_regular", "zigzag_boolean", "zigzag_cycle"}:
        implementation_certified = False
    theory_aligned = bool(method == "zigzag_certified" and implementation_certified)
    return {
        "certified": theory_aligned,
        "graph_certified": graph_certified,
        "implementation_certified": implementation_certified,
        "theory_aligned_method": theory_aligned,
    }

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

def _add_local_counts(rows: list[Counter[int]], block_size: int) -> None:
    for src, counts in enumerate(rows):
        block_start = (src // block_size) * block_size
        for dst in range(block_start, block_start + block_size):
            counts[dst] += 1

def _remote_counts_from_edges(seq_len: int, edges: list[tuple[int, int]]) -> list[Counter[int]]:
    rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    for src, dst in edges:
        rows[int(src)][int(dst)] += 1
    return rows

def build_method_counts(method: str, seq_len: int, args) -> list[Counter[int]] | None:
    method = canonical_method(method)
    if method == "dense":
        return None
    rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    _add_local_counts(rows, args.block_size)
    if method == "local":
        return rows
    if method == "random_regular":
        for src, dst in build_random_regular_cross_edges(
            seq_len, args.block_size, args.degree, args.seed
        ):
            rows[src][dst] += 1
        return rows
    if method == "zigzag_cycle":
        graph_config = DEFAULT_GRAPH_CONFIG
        remote_rows = build_zigzag_multiplicity(
            seq_len, args.block_size, args.degree, graph_config, include_local=False
        )
    elif method in {"zigzag_certified", "zigzag_boolean"}:
        graph_config = getattr(args, "graph_config", None)
        if graph_config is None:
            raise ValueError(f"{method} requires a graph artifact")
        remote_rows = build_zigzag_multiplicity(
            seq_len, args.block_size, args.degree, graph_config, include_local=False
        )
    else:
        raise ValueError(f"unknown method: {method}")
    for src, counts in enumerate(remote_rows):
        for dst, multiplicity in counts.items():
            if method in {"zigzag_boolean", "zigzag_cycle"}:
                rows[src][dst] = 1
            else:
                rows[src][dst] += int(multiplicity)
    if method == "zigzag_boolean":
        for src, counts in enumerate(rows):
            for dst in list(counts.keys()):
                rows[src][dst] = 1
    return rows

def counts_to_log_m_matrix(
    rows: list[Counter[int]],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    log_m = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    for src, counts in enumerate(rows):
        if counts:
            dst = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
            values = torch.tensor(
                [math.log(float(counts[int(key)])) for key in dst.tolist()],
                dtype=torch.float32,
                device=device,
            )
            log_m[src, dst] = values
    return log_m

def local_log_m_from_matrix(log_m: torch.Tensor, block_size: int) -> torch.Tensor:
    seq_len = log_m.shape[0]
    offsets = torch.arange(block_size, device=log_m.device)
    block_starts = (torch.arange(seq_len, device=log_m.device) // block_size) * block_size
    local_positions = block_starts[:, None] + offsets[None, :]
    return log_m.gather(1, local_positions)

def metrics_from_counts(
    rows: list[Counter[int]] | None,
    mask: torch.Tensor,
    method: str,
    block_size: int,
    degree: int,
) -> dict:
    metric = mask_metrics(mask, method, block_size, degree)
    if rows is None:
        metric.update(
            {
                "pre_causal_unique_k_mean": float(mask.shape[1]),
                "pre_causal_pair_count": int(mask.numel()),
                "multiplicity_max": 1,
                "multiplicity_mean_nonzero": 1.0,
            }
        )
        return metric
    pre_unique = [len(counts) for counts in rows]
    mult_values = [count for counts in rows for count in counts.values()]
    metric.update(
        {
            "pre_causal_unique_k_mean": float(np.mean(pre_unique)) if pre_unique else 0.0,
            "pre_causal_pair_count": int(sum(pre_unique)),
            "multiplicity_max": int(max(mult_values)) if mult_values else 0,
            "multiplicity_mean_nonzero": float(np.mean(mult_values)) if mult_values else 0.0,
        }
    )
    return metric

@dataclass
class AttentionArtifacts:
    mask: torch.Tensor
    local_valid: torch.Tensor
    neighbors: torch.Tensor | None
    valid_neighbors: torch.Tensor | None
    block_pair_index: torch.Tensor | None
    local_log_m: torch.Tensor | None
    neighbor_log_m: torch.Tensor | None
    metrics: dict

def make_attention_artifacts(
    method: str,
    seq_len: int,
    args,
    device: torch.device,
    attention_backend: str,
) -> AttentionArtifacts:
    method = canonical_method(method)
    rows = build_method_counts(method, seq_len, args)
    if rows is None:
        structural_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        log_m_matrix = None
    else:
        structural_mask = counts_to_mask(rows, seq_len, device)
        log_m_matrix = counts_to_log_m_matrix(rows, seq_len, device).detach()
        if getattr(args, "multiplicity_mode", "boolean") != "unique_log_m" or method != "zigzag_certified":
            log_m_matrix = torch.zeros_like(log_m_matrix)
    causal_mask = build_causal_mask(seq_len, device) if args.causal else None
    mask = structural_mask & causal_mask if causal_mask is not None else structural_mask
    local_valid = local_valid_from_mask(mask, args.block_size)
    neighbors = None
    valid_neighbors = None
    block_pair_index = None
    local_log_m = None
    neighbor_log_m = None
    if attention_backend == "neighbor":
        neighbors, valid_neighbors = mask_to_neighbors(mask)
        if log_m_matrix is not None:
            neighbor_log_m = log_m_matrix.gather(1, neighbors).masked_fill(~valid_neighbors, 0.0)
    elif attention_backend in {"split", "blockpair"}:
        local_mask = build_local_mask(seq_len, args.block_size, device)
        cross_mask = mask & ~local_mask
        neighbors, valid_neighbors = mask_to_neighbors(cross_mask)
        if log_m_matrix is not None:
            local_log_m = local_log_m_from_matrix(log_m_matrix, args.block_size)
            neighbor_log_m = log_m_matrix.gather(1, neighbors).masked_fill(~valid_neighbors, 0.0)
        if attention_backend == "blockpair":
            block_pair_index = cross_neighbors_to_block_pair_index(
                neighbors, valid_neighbors, args.block_size
            )
    metric = metrics_from_counts(rows, mask, method, args.block_size, args.degree)
    return AttentionArtifacts(
        mask=mask,
        local_valid=local_valid,
        neighbors=neighbors,
        valid_neighbors=valid_neighbors,
        block_pair_index=block_pair_index,
        local_log_m=local_log_m,
        neighbor_log_m=neighbor_log_m,
        metrics=metric,
    )
