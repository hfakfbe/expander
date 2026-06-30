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
    expected_raw_k,
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

V07_RESULT_EXTRA_FIELDS = [
    "timestamp_utc",
    "python_version",
    "graph_generation_algorithm",
    "canonical_graph_dir",
    "canonical_graph_artifact_path",
    "canonical_graph_artifact_sha256",
    "canonical_graph_seed",
    "canonical_graph_generation_algorithm",
    "graph_generation_status",
    "graph_generation_attempts",
    "graph_artifact_path",
    "graph_generation_path",
    "graph_certificate_path",
    "graph_artifact_sha256",
    "graph_artifact_sha256_matches_canonical",
    "graph_certificate_sha256",
    "N_total",
    "copy_source_length",
    "rho_zigzag_bound",
    "rho_zigzag_certified",
    "rho_zigzag_exact",
    "rot_g_is_bijection",
    "P_G_row_stochastic_error",
    "P_G_col_stochastic_error",
    "P_H_row_stochastic_error",
    "P_H_col_stochastic_error",
    "collision_count_mean",
    "zigzag_actual_k_min_after_causal",
    "zigzag_actual_k_mean_after_causal",
    "zigzag_actual_k_max_after_causal",
    "zigzag_attention_pair_count_after_causal",
    "random_target_k_source",
    "random_actual_k_min_after_causal",
    "random_actual_k_mean_after_causal",
    "random_actual_k_max_after_causal",
    "random_attention_pair_count_after_causal",
    "random_k_alignment_error_mean",
    "random_k_alignment_error_max",
    "random_alignment_mode",
    "random_k_aligned_to_zigzag",
    "base_learning_rate",
    "lr_scheduler",
    "warmup_ratio",
    "warmup_steps",
    "min_lr_ratio",
    "min_learning_rate",
    "cosine_total_steps",
    "weight_decay",
    "grad_clip_norm",
    "checkpoint_every",
    "total_wall_time_sec",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "data_prep_wall_time_sec",
    "summary_path",
    "raw_config_snapshot_path",
    "resolved_config_snapshot_path",
]

for field in V07_RESULT_EXTRA_FIELDS:
    if field not in RESULT_FIELDS:
        RESULT_FIELDS.append(field)

def method_certification_fields(method: str, certificate: dict, multiplicity_mode: str) -> dict:
    method = canonical_method(method)
    graph_methods = {"zigzag_certified", "zigzag_certified_cosine", "zigzag_boolean"}
    graph_certified = (
        bool(certificate.get("rho_zigzag_certified", certificate.get("certified")))
        if method in graph_methods
        else ""
    )
    implementation_certified = ""
    if method in {"zigzag_certified", "zigzag_certified_cosine"}:
        implementation_certified = bool(graph_certified and multiplicity_mode == "unique_log_m")
    elif method in {"random_regular", "zigzag_boolean", "zigzag_cycle"}:
        implementation_certified = False
    theory_aligned = bool(method in {"zigzag_certified", "zigzag_certified_cosine"} and implementation_certified)
    return {
        "certified": theory_aligned,
        "graph_certified": graph_certified,
        "implementation_certified": implementation_certified,
        "theory_aligned_method": theory_aligned,
    }

def resolve_attention_backend(requested: str, method: str) -> str:
    method = canonical_method(method)
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
    if method == "zigzag_certified_cosine":
        method = "zigzag_certified"
    if method == "dense":
        return None
    rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    add_local = not (
        method == "random_regular"
        and not bool(getattr(args, "random_include_local_edges", True))
    )
    if add_local:
        _add_local_counts(rows, args.block_size)
    if method == "local":
        return rows
    if method == "random_regular":
        target_rows = getattr(args, "random_aligned_rows", None)
        if target_rows is not None and len(target_rows) == seq_len:
            for src, counts in enumerate(target_rows):
                for dst, multiplicity in counts.items():
                    rows[src][int(dst)] += int(multiplicity)
        else:
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
    local_mask = build_local_mask(mask.shape[0], block_size, mask.device)
    local_pair_count = int((mask & local_mask).sum().item())
    attention_pair_count = int(mask.sum().item())
    metric.update(
        {
            "local_attention_pair_count": local_pair_count,
            "remote_attention_pair_count": attention_pair_count - local_pair_count,
            "local_attention_pair_rate": local_pair_count / max(attention_pair_count, 1),
        }
    )
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

def causal_row_k_from_counts(
    rows: list[Counter[int]] | None,
    seq_len: int,
    method: str,
    block_size: int,
    degree: int,
) -> dict:
    if rows is None:
        values = [src + 1 for src in range(seq_len)]
    else:
        values = [sum(1 for dst in counts if int(dst) <= src) for src, counts in enumerate(rows)]
    return {
        "raw_k": expected_raw_k(method, seq_len, block_size, degree),
        "actual_k_min_after_causal": int(min(values)) if values else 0,
        "actual_k_mean_after_causal": float(np.mean(values)) if values else 0.0,
        "actual_k_max_after_causal": int(max(values)) if values else 0,
        "attention_pair_count_after_causal": int(sum(values)),
        "per_query_k_after_causal": values,
    }

def build_random_rows_aligned_to_zigzag(seq_len: int, args) -> list[Counter[int]]:
    zigzag_rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    _add_local_counts(zigzag_rows, args.block_size)
    graph_config = getattr(args, "graph_config", None)
    if graph_config is None:
        raise ValueError("random_regular alignment requires a graph artifact")
    remote_rows = build_zigzag_multiplicity(
        seq_len, args.block_size, args.degree, graph_config, include_local=False
    )
    for src, counts in enumerate(remote_rows):
        for dst, multiplicity in counts.items():
            zigzag_rows[src][int(dst)] += int(multiplicity)

    import random

    rng = random.Random(
        f"random_aligned|{getattr(args, 'seed', 0)}|{seq_len}|{args.block_size}|{args.degree}"
    )
    random_rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    _add_local_counts(random_rows, args.block_size)
    for src, zigzag_counts in enumerate(zigzag_rows):
        target_total = sum(1 for dst in zigzag_counts if int(dst) <= src)
        local_causal = sum(1 for dst in random_rows[src] if int(dst) <= src)
        remote_target = max(0, target_total - local_causal)
        candidates = [dst for dst in range(src + 1) if dst not in random_rows[src]]
        remote_target = min(remote_target, len(candidates))
        for dst in rng.sample(candidates, remote_target):
            random_rows[src][dst] += 1
    return random_rows

def build_random_remote_rows_aligned_to_zigzag_noncausal(seq_len: int, args) -> list[Counter[int]]:
    """Remote random rows whose non-causal unique K matches zigzag per query."""
    zigzag_rows = build_method_counts("zigzag_certified", seq_len, args)
    if zigzag_rows is None:
        raise ValueError("zigzag rows are required for random budget alignment")

    import random

    rng = random.Random(
        f"random_aligned_noncausal|{getattr(args, 'seed', 0)}|{seq_len}|{args.block_size}|{args.degree}"
    )
    random_remote_rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    block_size = int(args.block_size)
    for src, zigzag_counts in enumerate(zigzag_rows):
        block_start = (src // block_size) * block_size
        local_keys = set(range(block_start, block_start + block_size))
        local_unique = len(local_keys)
        target_total = len(zigzag_counts)
        remote_target = max(0, target_total - local_unique)
        candidates = [dst for dst in range(seq_len) if dst not in local_keys]
        if remote_target > len(candidates):
            raise ValueError(
                f"cannot align random non-causal K for row {src}: "
                f"need {remote_target}, have {len(candidates)}"
            )
        for dst in rng.sample(candidates, remote_target):
            random_remote_rows[src][dst] += 1
    return random_remote_rows

def build_random_remote_rows_for_actual_mask_density(
    seq_len: int,
    args,
    density: float,
) -> list[Counter[int]]:
    """Remote random rows whose final non-causal boolean mask has target density.

    build_method_counts("random_regular", ...) always adds the block-local
    mask first. This helper therefore returns only the remote rows, sampling
    unique non-local keys so that local + remote contains exactly
    round(density * seq_len * seq_len) true entries.
    """
    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"random actual mask density must be in [0, 1], got {density}")
    block_size = int(args.block_size)
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    local_pair_count = seq_len * block_size
    target_pair_count = int(round(density * seq_len * seq_len))
    max_pair_count = seq_len * seq_len
    if target_pair_count < local_pair_count:
        raise ValueError(
            f"requested density={density} gives {target_pair_count} pairs, "
            f"below required local mask pairs={local_pair_count}"
        )
    if target_pair_count > max_pair_count:
        raise ValueError(
            f"requested density={density} gives {target_pair_count} pairs, "
            f"above full mask pairs={max_pair_count}"
        )

    remote_pair_count = target_pair_count - local_pair_count
    max_remote_per_row = seq_len - block_size
    base_remote_k, extra_rows = divmod(remote_pair_count, seq_len)
    if base_remote_k > max_remote_per_row or (
        base_remote_k == max_remote_per_row and extra_rows
    ):
        raise ValueError(
            f"requested density={density} requires too many remote keys per row: "
            f"base={base_remote_k}, extra_rows={extra_rows}, max={max_remote_per_row}"
        )

    import random

    rng = random.Random(
        "random_actual_mask_density|"
        f"{getattr(args, 'seed', 0)}|{seq_len}|{block_size}|"
        f"{target_pair_count}|{density:.12g}|"
        f"layer={getattr(args, 'random_layer_index', 'shared')}"
    )
    random_remote_rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    for src in range(seq_len):
        block_start = (src // block_size) * block_size
        local_keys = set(range(block_start, block_start + block_size))
        remote_k = base_remote_k + (1 if src < extra_rows else 0)
        candidates = [dst for dst in range(seq_len) if dst not in local_keys]
        if remote_k > len(candidates):
            raise ValueError(
                f"cannot sample random actual-density row {src}: "
                f"need {remote_k}, have {len(candidates)}"
            )
        for dst in rng.sample(candidates, remote_k):
            random_remote_rows[src][dst] += 1
    return random_remote_rows

def build_pure_random_rows_for_actual_mask_density(
    seq_len: int,
    args,
    density: float,
    *,
    exclude_block_local: bool = True,
) -> list[Counter[int]]:
    """Pure random rows whose boolean mask alone has target density.

    Unlike build_random_remote_rows_for_actual_mask_density, this helper is
    used with random_include_local_edges=false, so no deterministic block-local
    mask is added by build_method_counts.  The configured density is therefore
    the number of sampled random edges divided by seq_len**2.
    """
    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"random actual mask density must be in [0, 1], got {density}")
    block_size = int(args.block_size)
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    target_pair_count = int(round(density * seq_len * seq_len))
    candidates_per_row = seq_len - block_size if exclude_block_local else seq_len
    max_pair_count = seq_len * candidates_per_row
    if target_pair_count > max_pair_count:
        raise ValueError(
            f"requested density={density} gives {target_pair_count} pairs, "
            f"above pure-random candidate pairs={max_pair_count}"
        )
    base_k, extra_rows = divmod(target_pair_count, seq_len)
    if base_k > candidates_per_row or (base_k == candidates_per_row and extra_rows):
        raise ValueError(
            f"requested density={density} requires too many random keys per row: "
            f"base={base_k}, extra_rows={extra_rows}, max={candidates_per_row}"
        )

    import random

    rng = random.Random(
        "pure_random_actual_mask_density|"
        f"{getattr(args, 'seed', 0)}|{seq_len}|{block_size}|"
        f"{target_pair_count}|{density:.12g}|"
        f"exclude_block_local={bool(exclude_block_local)}|"
        f"layer={getattr(args, 'random_layer_index', 'shared')}"
    )
    rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    for src in range(seq_len):
        random_k = base_k + (1 if src < extra_rows else 0)
        if exclude_block_local:
            block_start = (src // block_size) * block_size
            local_keys = set(range(block_start, block_start + block_size))
            candidates = [dst for dst in range(seq_len) if dst not in local_keys]
        else:
            candidates = list(range(seq_len))
        if random_k > len(candidates):
            raise ValueError(
                f"cannot sample pure random actual-density row {src}: "
                f"need {random_k}, have {len(candidates)}"
            )
        for dst in rng.sample(candidates, random_k):
            rows[src][dst] += 1
    return rows

def build_random_remote_rows_for_multihop_copy_route(
    seq_len: int,
    args,
    density: float,
) -> list[Counter[int]]:
    """Random actual-density rows with a guaranteed multi-hop Copy route.

    The corrected non-causal Copy task supervises positions source_len..T-1
    to reproduce positions 0..source_len-1. A pure random 10% mask leaves most
    target rows without a direct edge to the matching source, so the model has
    to rely on multi-layer propagation. This helper makes that propagation
    structurally learnable while preserving the configured boolean density:

    * every row keeps the usual block-local mask outside this helper;
    * each remote row receives one high-multiplicity route edge src -> src-s
      when src >= s, or, with random_route_layerwise_staged=true, each layer
      receives only its own stage of a source->marker-lane->target route;
    * random remote edges fill the remaining unique-key budget up to the exact
      requested actual mask density;
    * direct target-to-source Copy edges source_len+i -> i are excluded from
      the random filler so success cannot collapse to one-hop copying.

    With source_len=1024, layers=8, and the default
    route_stride=source_len/layers=128, every supervised target position has a
    deterministic 8-hop route to its matching source:

        1024+i -> 896+i -> ... -> i.

    The route multiplicity is consumed only as a log-m attention bias by callers
    that explicitly enable args.random_use_log_m; it does not increase the
    boolean density.
    """
    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"random actual mask density must be in [0, 1], got {density}")
    block_size = int(args.block_size)
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")

    source_len = int(getattr(args, "random_route_source_length", seq_len // 2))
    target_start = int(getattr(args, "random_route_target_start", source_len))
    layers = int(getattr(args, "random_route_layers", 1))
    if layers <= 0:
        raise ValueError(f"random multihop route requires positive layers, got {layers}")
    route_stride = int(getattr(args, "random_route_stride", 0))
    if route_stride <= 0:
        if source_len % layers != 0:
            raise ValueError(
                f"source_len={source_len} must be divisible by layers={layers} "
                "when random_route_stride is not configured"
            )
        route_stride = source_len // layers
    if route_stride <= 0:
        raise ValueError(f"random multihop route stride must be positive, got {route_stride}")
    if target_start + source_len > seq_len:
        raise ValueError(
            f"copy route target window [{target_start}, {target_start + source_len}) "
            f"escapes seq_len={seq_len}"
        )
    expected_source = target_start - layers * route_stride
    if expected_source != 0:
        raise ValueError(
            "random multihop copy route expects target_start - layers*stride == 0, "
            f"got target_start={target_start}, layers={layers}, stride={route_stride}"
        )

    route_multiplicity = int(getattr(args, "random_route_multiplicity", 1))
    if route_multiplicity <= 0:
        raise ValueError(
            f"random multihop route multiplicity must be positive, got {route_multiplicity}"
        )

    local_pair_count = seq_len * block_size
    target_pair_count = int(round(density * seq_len * seq_len))
    max_pair_count = seq_len * seq_len
    if target_pair_count < local_pair_count:
        raise ValueError(
            f"requested density={density} gives {target_pair_count} pairs, "
            f"below required local mask pairs={local_pair_count}"
        )
    if target_pair_count > max_pair_count:
        raise ValueError(
            f"requested density={density} gives {target_pair_count} pairs, "
            f"above full mask pairs={max_pair_count}"
        )
    remote_pair_count = target_pair_count - local_pair_count
    max_remote_per_row = seq_len - block_size
    base_remote_k, extra_rows = divmod(remote_pair_count, seq_len)
    if base_remote_k > max_remote_per_row or (
        base_remote_k == max_remote_per_row and extra_rows
    ):
        raise ValueError(
            f"requested density={density} requires too many remote keys per row: "
            f"base={base_remote_k}, extra_rows={extra_rows}, max={max_remote_per_row}"
        )

    import random

    rng = random.Random(
        "random_multihop_copy_route|"
        f"{getattr(args, 'seed', 0)}|{seq_len}|{block_size}|"
        f"{target_pair_count}|{density:.12g}|"
        f"layers={layers}|stride={route_stride}|source={source_len}|"
        f"target_start={target_start}|"
        f"layer={getattr(args, 'random_layer_index', 'shared')}"
    )
    staged = bool(getattr(args, "random_route_layerwise_staged", False))
    layer_index = int(getattr(args, "random_layer_index", 0))
    route_keys_by_row: list[set[int]] = [set() for _ in range(seq_len)]
    if staged:
        if not 0 <= layer_index < layers:
            raise ValueError(f"random_layer_index must be in [0,{layers}), got {layer_index}")

        def stage_position(stage: int, offset: int) -> int:
            if stage <= 0:
                return int(offset)
            if stage >= layers:
                return target_start + int(offset)
            return target_start + ((int(offset) + stage * route_stride) % source_len)

        for offset in range(source_len):
            src = stage_position(layer_index + 1, offset)
            dst = stage_position(layer_index, offset)
            if not (0 <= src < seq_len and 0 <= dst < seq_len):
                raise ValueError(f"staged route produced invalid edge {src}->{dst}")
            block_start = (src // block_size) * block_size
            local_keys = set(range(block_start, block_start + block_size))
            if dst in local_keys:
                continue
            route_keys_by_row[src].add(dst)
    else:
        for src in range(seq_len):
            block_start = (src // block_size) * block_size
            local_keys = set(range(block_start, block_start + block_size))
            route_dst = src - route_stride
            if route_dst >= 0 and route_dst not in local_keys:
                route_keys_by_row[src].add(route_dst)

    random_remote_rows: list[Counter[int]] = [Counter() for _ in range(seq_len)]
    route_edges = 0
    direct_copy_edges_excluded = 0
    for src in range(seq_len):
        block_start = (src // block_size) * block_size
        local_keys = set(range(block_start, block_start + block_size))
        remote_k = base_remote_k + (1 if src < extra_rows else 0)
        route_keys = sorted(route_keys_by_row[src])
        if route_keys:
            if remote_k <= 0:
                raise ValueError(
                    f"row {src} has no remote budget for required route edge"
                )
            if len(route_keys) > remote_k:
                raise ValueError(
                    f"row {src} required route edges exceed remote budget: "
                    f"required={len(route_keys)}, budget={remote_k}"
                )
            for route_dst in route_keys:
                random_remote_rows[src][route_dst] += route_multiplicity
            route_edges += len(route_keys)

        forbidden: set[int] = set()
        target_offset = src - target_start
        if 0 <= target_offset < source_len:
            forbidden.add(target_offset)
        candidates = [
            dst
            for dst in range(seq_len)
            if dst not in local_keys
            and dst not in random_remote_rows[src]
            and dst not in forbidden
        ]
        filler_k = remote_k - len(random_remote_rows[src])
        if filler_k < 0:
            raise ValueError(
                f"row {src} required route edges exceed remote budget: "
                f"required={len(random_remote_rows[src])}, budget={remote_k}"
            )
        if filler_k > len(candidates):
            raise ValueError(
                f"cannot sample random multihop route row {src}: "
                f"need {filler_k}, have {len(candidates)}"
            )
        for dst in rng.sample(candidates, filler_k):
            random_remote_rows[src][dst] += 1
        if forbidden and forbidden.isdisjoint(random_remote_rows[src].keys()):
            direct_copy_edges_excluded += 1

    setattr(args, "random_multihop_route_stride", int(route_stride))
    setattr(args, "random_multihop_route_edges", int(route_edges))
    setattr(args, "random_multihop_route_layerwise_staged", bool(staged))
    setattr(args, "random_multihop_route_layer_index", int(layer_index) if staged else None)
    setattr(args, "random_multihop_direct_copy_edges_excluded", int(direct_copy_edges_excluded))
    return random_remote_rows

def budget_diagnostics(seq_len: int, args) -> tuple[dict, dict, list[Counter[int]]]:
    zigzag_rows = build_method_counts("zigzag_certified", seq_len, args)
    random_rows = build_random_rows_aligned_to_zigzag(seq_len, args)
    zigzag = causal_row_k_from_counts(
        zigzag_rows, seq_len, "zigzag_certified", args.block_size, args.degree
    )
    random_diag = causal_row_k_from_counts(
        random_rows, seq_len, "random_regular", args.block_size, args.degree
    )
    zigzag_k = zigzag["per_query_k_after_causal"]
    random_k = random_diag["per_query_k_after_causal"]
    errors = [abs(int(a) - int(b)) for a, b in zip(random_k, zigzag_k)]
    mode = getattr(args, "random_alignment_mode", "per_query")
    zigzag.update(
        {
            "random_target_k_source": "zigzag_actual_post_causal",
            "random_alignment_mode": mode,
        }
    )
    random_diag.update(
        {
            "random_target_k_source": "zigzag_actual_post_causal",
            "random_alignment_mode": mode,
            "random_k_alignment_error_mean": float(np.mean(errors)) if errors else 0.0,
            "random_k_alignment_error_max": int(max(errors)) if errors else 0,
            "random_k_aligned_to_zigzag": bool(max(errors) == 0) if errors else True,
        }
    )
    return zigzag, random_diag, random_rows

@dataclass
class AttentionArtifacts:
    mask: torch.Tensor
    local_valid: torch.Tensor
    neighbors: torch.Tensor | None
    valid_neighbors: torch.Tensor | None
    block_pair_index: torch.Tensor | None
    local_log_m: torch.Tensor | None
    neighbor_log_m: torch.Tensor | None
    route_transport_src: torch.Tensor | None
    route_transport_dst: torch.Tensor | None
    route_transport_scale: float | None
    route_transport_mode: str | None
    metrics: dict


def _to_device_if_tensor(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device_if_tensor(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device_if_tensor(item, device) for item in value)
    return value


def attention_artifacts_to_device(artifacts: AttentionArtifacts, device: torch.device) -> AttentionArtifacts:
    return AttentionArtifacts(
        mask=_to_device_if_tensor(artifacts.mask, device),
        local_valid=_to_device_if_tensor(artifacts.local_valid, device),
        neighbors=_to_device_if_tensor(artifacts.neighbors, device),
        valid_neighbors=_to_device_if_tensor(artifacts.valid_neighbors, device),
        block_pair_index=_to_device_if_tensor(artifacts.block_pair_index, device),
        local_log_m=_to_device_if_tensor(artifacts.local_log_m, device),
        neighbor_log_m=_to_device_if_tensor(artifacts.neighbor_log_m, device),
        route_transport_src=_to_device_if_tensor(artifacts.route_transport_src, device),
        route_transport_dst=_to_device_if_tensor(artifacts.route_transport_dst, device),
        route_transport_scale=artifacts.route_transport_scale,
        route_transport_mode=artifacts.route_transport_mode,
        metrics=artifacts.metrics,
    )

def make_attention_artifacts(
    method: str,
    seq_len: int,
    args,
    device: torch.device,
    attention_backend: str,
) -> AttentionArtifacts:
    method = canonical_method(method)
    if method == "zigzag_certified_cosine":
        method = "zigzag_certified"
    rows = build_method_counts(method, seq_len, args)
    if rows is None:
        structural_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        log_m_matrix = None
    else:
        structural_mask = counts_to_mask(rows, seq_len, device)
        log_m_matrix = counts_to_log_m_matrix(rows, seq_len, device).detach()
        use_zigzag_log_m = method == "zigzag_certified"
        use_random_log_m = method == "random_regular" and bool(getattr(args, "random_use_log_m", False))
        if getattr(args, "multiplicity_mode", "boolean") != "unique_log_m" or not (use_zigzag_log_m or use_random_log_m):
            log_m_matrix = torch.zeros_like(log_m_matrix)
    causal_mask = build_causal_mask(seq_len, device) if args.causal else None
    mask = structural_mask & causal_mask if causal_mask is not None else structural_mask
    local_valid = local_valid_from_mask(mask, args.block_size)
    neighbors = None
    valid_neighbors = None
    block_pair_index = None
    local_log_m = None
    neighbor_log_m = None
    route_transport_src = None
    route_transport_dst = None
    route_transport_scale = None
    route_transport_mode = None
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
    if method == "random_regular" and bool(getattr(args, "random_multihop_copy_route", False)):
        source_len = int(getattr(args, "random_route_source_length", seq_len // 2))
        target_start = int(getattr(args, "random_route_target_start", source_len))
        layers = int(getattr(args, "random_route_layers", 1))
        stride = int(getattr(args, "random_multihop_route_stride", getattr(args, "random_route_stride", 0)))
        route_multiplicity = int(getattr(args, "random_route_multiplicity", 1))
        route_edge_count = 0
        route_edge_missing = 0
        one_hop_direct = 0
        transport_src: list[int] = []
        transport_dst: list[int] = []
        if rows is not None:
            staged = bool(getattr(args, "random_route_layerwise_staged", False))
            if staged:
                layer_index = int(getattr(args, "random_layer_index", 0))

                def stage_position(stage: int, offset: int) -> int:
                    if stage <= 0:
                        return int(offset)
                    if stage >= layers:
                        return target_start + int(offset)
                    return target_start + ((int(offset) + stage * stride) % source_len)

                for offset in range(source_len):
                    src = stage_position(layer_index + 1, offset)
                    route_dst = stage_position(layer_index, offset)
                    if (src // int(args.block_size)) == (route_dst // int(args.block_size)):
                        continue
                    if int(rows[src].get(route_dst, 0)) >= route_multiplicity:
                        route_edge_count += 1
                        transport_src.append(int(src))
                        transport_dst.append(int(route_dst))
                    else:
                        route_edge_missing += 1
            else:
                for src, counts in enumerate(rows):
                    route_dst = src - stride
                    if stride > 0 and route_dst >= 0:
                        if int(counts.get(route_dst, 0)) >= route_multiplicity:
                            route_edge_count += 1
                            transport_src.append(int(src))
                            transport_dst.append(int(route_dst))
                        else:
                            route_edge_missing += 1
            for offset in range(source_len):
                target = target_start + offset
                if 0 <= target < len(rows) and int(rows[target].get(offset, 0)) > 0:
                    one_hop_direct += 1
        if bool(getattr(args, "random_route_transport", False)):
            if route_edge_missing:
                raise ValueError("random_route_transport requires all route edges to be present")
            if transport_src:
                route_transport_src = torch.tensor(transport_src, dtype=torch.long, device=device)
                route_transport_dst = torch.tensor(transport_dst, dtype=torch.long, device=device)
            else:
                route_transport_src = torch.empty((0,), dtype=torch.long, device=device)
                route_transport_dst = torch.empty((0,), dtype=torch.long, device=device)
            route_transport_scale = float(getattr(args, "random_route_transport_scale", 1.0))
            route_transport_mode = str(getattr(args, "random_route_transport_mode", "residual"))
            if route_transport_mode not in {"residual", "replace", "memory_residual", "memory_replace"}:
                raise ValueError(f"unknown random_route_transport_mode={route_transport_mode!r}")
        metric.update(
            {
                "random_multihop_copy_route": True,
                "random_route_layers": layers,
                "random_route_stride": stride,
                "random_route_source_length": source_len,
                "random_route_target_start": target_start,
                "random_route_multiplicity": route_multiplicity,
                "random_route_edge_count": route_edge_count,
                "random_route_edge_missing": route_edge_missing,
                "random_direct_copy_edge_count": one_hop_direct,
                "random_direct_copy_edge_rate": one_hop_direct / max(source_len, 1),
                "random_use_log_m": bool(getattr(args, "random_use_log_m", False)),
                "random_route_layerwise_staged": bool(getattr(args, "random_route_layerwise_staged", False)),
                "random_route_layer_index": getattr(args, "random_layer_index", None)
                if bool(getattr(args, "random_route_layerwise_staged", False))
                else None,
                "random_route_transport": bool(getattr(args, "random_route_transport", False)),
                "random_route_transport_edge_count": len(transport_src),
                "random_route_transport_scale": route_transport_scale,
                "random_route_transport_mode": route_transport_mode,
            }
        )
    return AttentionArtifacts(
        mask=mask,
        local_valid=local_valid,
        neighbors=neighbors,
        valid_neighbors=valid_neighbors,
        block_pair_index=block_pair_index,
        local_log_m=local_log_m,
        neighbor_log_m=neighbor_log_m,
        route_transport_src=route_transport_src,
        route_transport_dst=route_transport_dst,
        route_transport_scale=route_transport_scale,
        route_transport_mode=route_transport_mode,
        metrics=metric,
    )
