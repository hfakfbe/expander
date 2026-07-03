from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from probe_common import command_string, file_sha256, git_commit, git_dirty, read_json, write_command, write_json, write_jsonl
from probe_metrics import aggregate_metric_rows, json_metric, masked_sequence_loss_sum, sequence_metrics
from probe_tasks import JsonlStore, ProbeTransformer, load_encoder, make_probe_batch, parameter_count
from runtime_common import (
    epoch_coverage,
    select_device,
    state_dict_sha256,
    tensor_checkpoint_sha256,
    write_checkpoint_manifest,
)
from run_probe_experiment import schedule_lr
from synthetic_mvp_core.artifacts import (
    attention_artifacts_to_device,
    build_pure_random_rows_for_actual_mask_density,
    build_random_remote_rows_aligned_to_zigzag_noncausal,
    build_random_remote_rows_for_actual_mask_density,
    build_random_remote_rows_for_multihop_copy_route,
    make_attention_artifacts,
    resolve_attention_backend,
)
from graph_structures import build_graph_artifact


VERSION = "copy_corrected_v01_l8_log5"
BRANCH = "codex/copy-corrected-v01-l8-log5"


def install_numpy_pickle_compat_aliases() -> None:
    """Allow old/new NumPy RNG pickles to load across NumPy 1.x/2.x module names."""
    if "numpy._core" not in sys.modules:
        try:
            import numpy.core as numpy_core

            sys.modules["numpy._core"] = numpy_core
        except Exception:
            pass
    if "numpy._core.multiarray" not in sys.modules:
        try:
            import numpy.core.multiarray as numpy_core_multiarray

            sys.modules["numpy._core.multiarray"] = numpy_core_multiarray
        except Exception:
            pass
    if "numpy._core.numeric" not in sys.modules:
        try:
            import numpy.core.numeric as numpy_core_numeric

            sys.modules["numpy._core.numeric"] = numpy_core_numeric
        except Exception:
            pass


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    install_numpy_pickle_compat_aliases()
    return torch.load(path, map_location=device, weights_only=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_all_seeds(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_algorithms = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "python_random_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "torch_cuda_manual_seed_all": bool(torch.cuda.is_available()),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": deterministic_algorithms,
    }


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return read_json(Path(config["task_parameter_manifest"]))


def task_record(manifest: dict[str, Any]) -> dict[str, Any]:
    records = [row for row in manifest["tasks"] if row["task"] == "copy"]
    if len(records) != 1:
        raise ValueError("copy_corrected_v01 manifest must contain exactly one copy task")
    record = records[0]
    if not bool(record.get("copy_corrected_v01")):
        raise ValueError("manifest task is not marked copy_corrected_v01")
    return record


def deterministic_permutation(n: int, data_seed: int, epoch: int) -> list[int]:
    indices = list(range(int(n)))
    rng = random.Random(f"copy_corrected_v01|data|{int(data_seed)}|epoch|{int(epoch)}")
    rng.shuffle(indices)
    return indices


def run_dir_for(config: dict[str, Any], method: str, seed: int, mode: str = "train") -> Path:
    trial = str(config.get("trial_id", "gate"))
    if mode == "gate-overfit":
        return Path(config["output_root"]) / trial / "gate_overfit" / method / f"seed{int(seed)}"
    return Path(config["output_root"]) / trial / method / f"seed{int(seed)}"


def experiment_version(config: dict[str, Any], record: dict[str, Any]) -> str:
    return str(config.get("version") or record.get("copy_corrected_variant") or VERSION)


def experiment_branch(record: dict[str, Any]) -> str:
    return str(record.get("branch_name") or BRANCH)


def run_id_for(config: dict[str, Any], record: dict[str, Any], method: str, seed: int, suffix: str = "") -> str:
    base = f"{experiment_version(config, record)}_{config.get('trial_id', 'gate')}_{method}_seed{seed}"
    return f"{base}_{suffix}" if suffix else base


def config_random_density(config: dict[str, Any]) -> Any:
    density = config.get("random_actual_mask_density")
    if density is None:
        density = dict(config.get("attention", {})).get("random_actual_mask_density")
    return density


def config_random_layerwise_independent(config: dict[str, Any]) -> bool:
    value = config.get("random_layerwise_independent_masks")
    if value is None:
        value = dict(config.get("attention", {})).get("random_layerwise_independent_masks", False)
    return bool(value)


def config_random_include_local_edges(config: dict[str, Any]) -> bool:
    value = config.get("random_include_local_edges")
    if value is None:
        value = dict(config.get("attention", {})).get("random_include_local_edges", True)
    return bool(value)


def config_layerwise_zigzag_random_graphs(config: dict[str, Any]) -> bool:
    value = config.get("zigzag_layerwise_random_graphs")
    if value is None:
        value = dict(config.get("attention", {})).get("zigzag_layerwise_random_graphs", False)
    return bool(value)


def config_random_multihop_copy_route(config: dict[str, Any]) -> bool:
    value = config.get("random_multihop_copy_route")
    if value is None:
        value = dict(config.get("attention", {})).get("random_multihop_copy_route", False)
    return bool(value)


def config_random_route_multiplicity(config: dict[str, Any]) -> int:
    value = config.get("random_route_multiplicity")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_multiplicity", 65536)
    return int(value)


def config_random_route_use_log_m(config: dict[str, Any]) -> bool:
    value = config.get("random_route_use_log_m")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_use_log_m", True)
    return bool(value)


def config_random_route_stride(config: dict[str, Any]) -> int | None:
    value = config.get("random_route_stride")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_stride")
    return None if value is None else int(value)


def config_random_route_layerwise_staged(config: dict[str, Any]) -> bool:
    value = config.get("random_route_layerwise_staged")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_layerwise_staged", False)
    return bool(value)


def config_random_route_transport(config: dict[str, Any]) -> bool:
    value = config.get("random_route_transport")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_transport", False)
    return bool(value)


def config_random_route_transport_scale(config: dict[str, Any]) -> float:
    value = config.get("random_route_transport_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_transport_scale", 1.0)
    return float(value)


def config_random_route_transport_mode(config: dict[str, Any]) -> str:
    value = config.get("random_route_transport_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_route_transport_mode", "residual")
    value = str(value)
    if value not in {"residual", "replace", "memory_residual", "memory_replace"}:
        raise ValueError(f"unknown random_route_transport_mode={value!r}")
    return value


def config_random_learned_attention_edge_bias(config: dict[str, Any]) -> bool:
    value = config.get("random_learned_attention_edge_bias")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_attention_edge_bias", False)
    return bool(value)


def config_random_learned_edge_memory_transport_mode(config: dict[str, Any]) -> str | None:
    value = config.get("random_learned_edge_memory_transport_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_edge_memory_transport_mode")
    if value is None:
        return None
    value = str(value)
    if value == "none":
        return None
    if value not in {"residual", "replace", "residual_update"}:
        raise ValueError(f"unknown random_learned_edge_memory_transport_mode={value!r}")
    return value


def config_random_learned_edge_memory_transport_scale(config: dict[str, Any]) -> float:
    value = config.get("random_learned_edge_memory_transport_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_edge_memory_transport_scale", 1.0)
    return float(value)


def config_random_learned_edge_memory_transport_temperature(config: dict[str, Any]) -> float:
    value = config.get("random_learned_edge_memory_transport_temperature")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_edge_memory_transport_temperature", 1.0)
    value = float(value)
    if value <= 0:
        raise ValueError("random_learned_edge_memory_transport_temperature must be positive")
    return value


def config_random_learned_edge_bias_init(config: dict[str, Any]) -> float:
    value = config.get("random_learned_edge_bias_init")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_edge_bias_init", 0.0)
    return float(value)


def config_random_value_position_encoding(config: dict[str, Any]) -> str:
    value = config.get("random_value_position_encoding")
    if value is None:
        value = dict(config.get("attention", {})).get("random_value_position_encoding", "none")
    value = str(value)
    if value not in {"none", "rope_relative"}:
        raise ValueError(f"unknown random_value_position_encoding={value!r}")
    return value


def config_random_relative_attention_bias(config: dict[str, Any]) -> bool:
    value = config.get("random_relative_attention_bias")
    if value is None:
        value = dict(config.get("attention", {})).get("random_relative_attention_bias", False)
    return bool(value)


def config_random_relative_attention_bias_init(config: dict[str, Any]) -> float:
    value = config.get("random_relative_attention_bias_init")
    if value is None:
        value = dict(config.get("attention", {})).get("random_relative_attention_bias_init", 0.0)
    return float(value)


def config_random_attention_logit_scale_multiplier(config: dict[str, Any]) -> float:
    value = config.get("random_attention_logit_scale_multiplier")
    if value is None:
        value = dict(config.get("attention", {})).get("random_attention_logit_scale_multiplier", 1.0)
    value = float(value)
    if value <= 0:
        raise ValueError("random_attention_logit_scale_multiplier must be positive")
    return value


def config_random_learned_attention_logit_scale(config: dict[str, Any]) -> bool:
    value = config.get("random_learned_attention_logit_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_learned_attention_logit_scale", False)
    return bool(value)


def config_random_attention_top_k(config: dict[str, Any]) -> int:
    value = config.get("random_attention_top_k")
    if value is None:
        value = dict(config.get("attention", {})).get("random_attention_top_k", 0)
    value = int(value)
    if value < 0:
        raise ValueError("random_attention_top_k must be non-negative")
    return value


def config_random_rollout_memory(config: dict[str, Any]) -> bool:
    value = config.get("random_rollout_memory")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory", False)
    return bool(value)


def config_random_rollout_memory_scale(config: dict[str, Any]) -> float:
    value = config.get("random_rollout_memory_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_scale", 1.0)
    return float(value)


def config_random_rollout_memory_source(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_memory_source")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_source", "input")
    value = str(value)
    if value not in {"input", "hidden"}:
        raise ValueError(f"unknown random_rollout_memory_source={value!r}")
    return value


def config_random_rollout_memory_update(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_memory_update")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_update", "replace")
    value = str(value)
    if value not in {"replace", "residual", "lazy"}:
        raise ValueError(f"unknown random_rollout_memory_update={value!r}")
    return value


def config_random_rollout_memory_lazy_alpha(config: dict[str, Any]) -> float:
    value = config.get("random_rollout_memory_lazy_alpha")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_lazy_alpha", 0.0)
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("random_rollout_memory_lazy_alpha must be in [0, 1]")
    return value


def config_random_rollout_memory_learned_update(config: dict[str, Any]) -> bool:
    value = config.get("random_rollout_memory_learned_update")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_learned_update", False)
    return bool(value)


def config_random_rollout_memory_learned_scale(config: dict[str, Any]) -> bool:
    value = config.get("random_rollout_memory_learned_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_learned_scale", False)
    return bool(value)


def config_random_rollout_memory_steps(config: dict[str, Any]) -> int:
    value = config.get("random_rollout_memory_steps")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_steps", 1)
    value = int(value)
    if value < 1:
        raise ValueError("random_rollout_memory_steps must be >= 1")
    return value


def config_random_rollout_memory_multiscale_steps(config: dict[str, Any]) -> list[int]:
    value = config.get("random_rollout_memory_multiscale_steps")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_multiscale_steps")
    if value is None:
        return [config_random_rollout_memory_steps(config)]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    steps = [int(step) for step in raw_values]
    if not steps:
        raise ValueError("random_rollout_memory_multiscale_steps must not be empty")
    if any(step < 1 for step in steps):
        raise ValueError("random_rollout_memory_multiscale_steps values must be >= 1")
    return steps


def config_random_rollout_memory_multiscale_weights(config: dict[str, Any]) -> list[float] | None:
    value = config.get("random_rollout_memory_multiscale_weights")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_memory_multiscale_weights")
    if value is None:
        return None
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    weights = [float(weight) for weight in raw_values]
    if not weights:
        raise ValueError("random_rollout_memory_multiscale_weights must not be empty")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("random_rollout_memory_multiscale_weights values must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("random_rollout_memory_multiscale_weights must have positive sum")
    if len(weights) != len(config_random_rollout_memory_multiscale_steps(config)):
        raise ValueError("random_rollout_memory_multiscale_weights length must match multiscale steps length")
    total = float(sum(weights))
    return [weight / total for weight in weights]


def config_random_rollout_head_merge(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_head_merge")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_head_merge", "mean")
    value = str(value)
    if value not in {"mean", "concat_linear"}:
        raise ValueError(f"unknown random_rollout_head_merge={value!r}")
    return value


def config_random_rollout_weight_mode(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_weight_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_weight_mode", "soft")
    value = str(value)
    if value not in {"soft", "straight_through_hard"}:
        raise ValueError(f"unknown random_rollout_weight_mode={value!r}")
    return value


def config_random_rollout_edge_scope(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_edge_scope")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_edge_scope", "all")
    value = str(value)
    if value not in {"all", "cross_only", "local_only"}:
        raise ValueError(f"unknown random_rollout_edge_scope={value!r}")
    return value


def config_random_rollout_local_logit_bias(config: dict[str, Any]) -> float:
    value = config.get("random_rollout_local_logit_bias")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_local_logit_bias", 0.0)
    return float(value)


def config_random_rollout_cross_logit_bias(config: dict[str, Any]) -> float:
    value = config.get("random_rollout_cross_logit_bias")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_cross_logit_bias", 0.0)
    return float(value)


def config_random_rollout_output_logits(config: dict[str, Any]) -> bool:
    value = config.get("random_rollout_output_logits")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_output_logits", False)
    return bool(value)


def config_random_rollout_output_scale(config: dict[str, Any]) -> float:
    value = config.get("random_rollout_output_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_output_scale", 1.0)
    return float(value)


def config_random_rollout_output_mode(config: dict[str, Any]) -> str:
    value = config.get("random_rollout_output_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_rollout_output_mode", "shared_head")
    value = str(value)
    if value not in {"shared_head", "separate_head", "embedding_tied", "embedding_tied_cosine"}:
        raise ValueError(f"unknown random_rollout_output_mode={value!r}")
    return value


def config_random_attention_residual_scale(config: dict[str, Any]) -> float:
    value = config.get("random_attention_residual_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_attention_residual_scale", 1.0)
    return float(value)


def config_random_ffn_residual_scale(config: dict[str, Any]) -> float:
    value = config.get("random_ffn_residual_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_ffn_residual_scale", 1.0)
    return float(value)


def config_random_history_output_logits(config: dict[str, Any]) -> bool:
    value = config.get("random_history_output_logits")
    if value is None:
        value = dict(config.get("attention", {})).get("random_history_output_logits", False)
    return bool(value)


def config_random_history_output_scale(config: dict[str, Any]) -> float:
    value = config.get("random_history_output_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_history_output_scale", 1.0)
    return float(value)


def config_random_history_output_source(config: dict[str, Any]) -> str:
    value = config.get("random_history_output_source")
    if value is None:
        value = dict(config.get("attention", {})).get("random_history_output_source", "hidden")
    value = str(value)
    if value not in {"hidden", "rollout", "hidden_rollout"}:
        raise ValueError(f"unknown random_history_output_source={value!r}")
    return value


def config_random_history_output_merge(config: dict[str, Any]) -> str:
    value = config.get("random_history_output_merge")
    if value is None:
        value = dict(config.get("attention", {})).get("random_history_output_merge", "concat")
    value = str(value)
    if value not in {"concat", "weighted_sum", "logit_weighted_sum", "confidence_logit_weighted_sum"}:
        raise ValueError(f"unknown random_history_output_merge={value!r}")
    return value


def config_random_history_include_input(config: dict[str, Any]) -> bool:
    value = config.get("random_history_include_input")
    if value is None:
        value = dict(config.get("attention", {})).get("random_history_include_input", True)
    return bool(value)


def config_random_positional_rollout_memory(config: dict[str, Any]) -> bool:
    value = config.get("random_positional_rollout_memory")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_memory", False)
    return bool(value)


def config_random_positional_rollout_scale(config: dict[str, Any]) -> float:
    value = config.get("random_positional_rollout_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_scale", 1.0)
    return float(value)


def config_random_positional_rollout_update(config: dict[str, Any]) -> str:
    value = config.get("random_positional_rollout_update")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_update", "replace")
    value = str(value)
    if value not in {"replace", "residual"}:
        raise ValueError(f"unknown random_positional_rollout_update={value!r}")
    return value


def config_random_positional_rollout_head_merge(config: dict[str, Any]) -> str:
    value = config.get("random_positional_rollout_head_merge")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_head_merge", "mean")
    value = str(value)
    if value not in {"mean", "concat_linear"}:
        raise ValueError(f"unknown random_positional_rollout_head_merge={value!r}")
    return value


def config_random_positional_rollout_output_logits(config: dict[str, Any]) -> bool:
    value = config.get("random_positional_rollout_output_logits")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_output_logits", False)
    return bool(value)


def config_random_positional_rollout_output_scale(config: dict[str, Any]) -> float:
    value = config.get("random_positional_rollout_output_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_output_scale", 1.0)
    return float(value)


def config_random_positional_rollout_output_mode(config: dict[str, Any]) -> str:
    value = config.get("random_positional_rollout_output_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_positional_rollout_output_mode", "shared_head")
    value = str(value)
    if value not in {"shared_head", "separate_head", "embedding_tied_cosine"}:
        raise ValueError(f"unknown random_positional_rollout_output_mode={value!r}")
    return value


def config_random_token_rollout_memory(config: dict[str, Any]) -> bool:
    value = config.get("random_token_rollout_memory")
    if value is None:
        value = dict(config.get("attention", {})).get("random_token_rollout_memory", False)
    return bool(value)


def config_random_token_rollout_scale(config: dict[str, Any]) -> float:
    value = config.get("random_token_rollout_scale")
    if value is None:
        value = dict(config.get("attention", {})).get("random_token_rollout_scale", 1.0)
    return float(value)


def config_random_token_rollout_logit_mode(config: dict[str, Any]) -> str:
    value = config.get("random_token_rollout_logit_mode")
    if value is None:
        value = dict(config.get("attention", {})).get("random_token_rollout_logit_mode", "prob")
    value = str(value)
    if value not in {"prob", "log"}:
        raise ValueError(f"unknown random_token_rollout_logit_mode={value!r}")
    return value


def config_resume_from_checkpoint(config: dict[str, Any]) -> str | None:
    value = config.get("resume_from_checkpoint")
    if value is None:
        value = dict(config.get("train", {})).get("resume_from_checkpoint")
    if value is None:
        return None
    value = str(value)
    return value or None


def config_resume_model_strict(config: dict[str, Any]) -> bool:
    value = config.get("resume_model_strict")
    if value is None:
        value = dict(config.get("train", {})).get("resume_model_strict", True)
    return bool(value)


def config_copy_margin_loss_weight(config: dict[str, Any]) -> float:
    value = config.get("copy_margin_loss_weight")
    if value is None:
        value = dict(config.get("train", {})).get("copy_margin_loss_weight", 0.0)
    value = float(value)
    if value < 0:
        raise ValueError("copy_margin_loss_weight must be non-negative")
    return value


def config_copy_margin_target(config: dict[str, Any]) -> float:
    value = config.get("copy_margin_target")
    if value is None:
        value = dict(config.get("train", {})).get("copy_margin_target", 1.0)
    return float(value)


def config_zigzag_graph_seed_base(config: dict[str, Any], default_seed: int = 0) -> int:
    value = config.get("zigzag_layerwise_graph_seed_base")
    if value is None:
        value = dict(config.get("attention", {})).get("zigzag_layerwise_graph_seed_base", default_seed)
    return int(value)


def run_identity(config_path: Path, config: dict[str, Any], manifest_path: Path, record: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    graph = record["graph_artifacts"]
    random_actual_mask_density = config_random_density(config)
    random_layerwise_independent = config_random_layerwise_independent(config)
    random_include_local_edges = config_random_include_local_edges(config)
    zigzag_layerwise_random_graphs = config_layerwise_zigzag_random_graphs(config)
    random_multihop_copy_route = config_random_multihop_copy_route(config)
    identity = {
        "version": experiment_version(config, record),
        "trial_id": config.get("trial_id", "gate"),
        "method": method,
        "seed": int(seed),
        "random_actual_mask_density": random_actual_mask_density if method == "random_regular" else None,
        "random_include_local_edges": random_include_local_edges if method == "random_regular" else None,
        "random_layerwise_independent_masks": random_layerwise_independent if method == "random_regular" else None,
        "random_layerwise_mask_count": int(record["resolved_layers"]) if method == "random_regular" and random_layerwise_independent else None,
        "random_multihop_copy_route": random_multihop_copy_route if method == "random_regular" else None,
        "random_route_multiplicity": config_random_route_multiplicity(config) if method == "random_regular" and random_multihop_copy_route else None,
        "random_route_use_log_m": config_random_route_use_log_m(config) if method == "random_regular" and random_multihop_copy_route else None,
        "random_route_stride": config_random_route_stride(config) if method == "random_regular" and random_multihop_copy_route else None,
        "random_route_layerwise_staged": config_random_route_layerwise_staged(config) if method == "random_regular" and random_multihop_copy_route else None,
        "random_route_transport": config_random_route_transport(config) if method == "random_regular" and random_multihop_copy_route else None,
        "random_route_transport_scale": config_random_route_transport_scale(config) if method == "random_regular" and random_multihop_copy_route and config_random_route_transport(config) else None,
        "random_route_transport_mode": config_random_route_transport_mode(config) if method == "random_regular" and random_multihop_copy_route and config_random_route_transport(config) else None,
        "random_learned_attention_edge_bias": config_random_learned_attention_edge_bias(config) if method == "random_regular" else None,
        "random_learned_edge_memory_transport_mode": config_random_learned_edge_memory_transport_mode(config) if method == "random_regular" else None,
        "random_learned_edge_memory_transport_scale": config_random_learned_edge_memory_transport_scale(config)
        if method == "random_regular" and config_random_learned_edge_memory_transport_mode(config)
        else None,
        "random_learned_edge_memory_transport_temperature": config_random_learned_edge_memory_transport_temperature(config)
        if method == "random_regular" and config_random_learned_edge_memory_transport_mode(config)
        else None,
        "random_learned_edge_bias_init": config_random_learned_edge_bias_init(config)
        if method == "random_regular"
        and (config_random_learned_attention_edge_bias(config) or config_random_learned_edge_memory_transport_mode(config))
        else None,
        "random_value_position_encoding": config_random_value_position_encoding(config) if method == "random_regular" else None,
        "zigzag_layerwise_random_graphs": zigzag_layerwise_random_graphs if method in {"zigzag_certified", "zigzag_boolean"} else None,
        "zigzag_layerwise_graph_seed_base": config_zigzag_graph_seed_base(config, seed) if method in {"zigzag_certified", "zigzag_boolean"} and zigzag_layerwise_random_graphs else None,
        "zigzag_layerwise_mask_count": int(record["resolved_layers"]) if method in {"zigzag_certified", "zigzag_boolean"} and zigzag_layerwise_random_graphs else None,
        "multiplicity_mode": "boolean" if method == "zigzag_boolean" else ("unique_log_m" if method == "zigzag_certified" else None),
        "branch_name": experiment_branch(record),
        "branch_head_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "resume_from_checkpoint": config_resume_from_checkpoint(config),
        "resume_model_strict": config_resume_model_strict(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": record["resolved_train_split_sha256"],
        "test_sha256": record["resolved_test_split_sha256"],
        "train_content_sha256": record["train_content_sha256"],
        "test_content_sha256": record["test_content_sha256"],
        "graph_sha256": graph["selected_graph_sha256"],
        "position_encoding": record["position_encoding"],
        "vocab_size": record["resolved_vocab_or_value_space_size"],
        "token_output_size": record["resolved_token_output_size"],
        "readout_start": record["resolved_readout_start"],
        "T": record["resolved_padded_sequence_length"],
    }
    if method == "random_regular" and config_random_relative_attention_bias(config):
        identity.update(
            {
                "random_relative_attention_bias": True,
                "random_relative_attention_bias_init": config_random_relative_attention_bias_init(config),
            }
        )
    if method == "random_regular" and (
        config_random_attention_logit_scale_multiplier(config) != 1.0
        or config_random_learned_attention_logit_scale(config)
        or config_random_attention_top_k(config) > 0
        or config_random_rollout_memory(config)
        or config_random_rollout_memory_learned_update(config)
        or config_random_rollout_memory_learned_scale(config)
        or config_random_rollout_output_logits(config)
        or config_random_attention_residual_scale(config) != 1.0
        or config_random_ffn_residual_scale(config) != 1.0
        or config_random_history_output_logits(config)
        or config_random_positional_rollout_memory(config)
        or config_random_positional_rollout_output_logits(config)
        or config_random_token_rollout_memory(config)
    ):
        identity.update(
            {
                "random_attention_logit_scale_multiplier": config_random_attention_logit_scale_multiplier(config),
                "random_learned_attention_logit_scale": config_random_learned_attention_logit_scale(config),
                "random_attention_top_k": config_random_attention_top_k(config),
                "random_rollout_memory": config_random_rollout_memory(config),
                "random_rollout_memory_scale": (
                    config_random_rollout_memory_scale(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_memory_source": (
                    config_random_rollout_memory_source(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_memory_update": (
                    config_random_rollout_memory_update(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_memory_lazy_alpha": (
                    config_random_rollout_memory_lazy_alpha(config)
                    if config_random_rollout_memory(config)
                    and config_random_rollout_memory_update(config) == "lazy"
                    else None
                ),
                "random_rollout_memory_learned_update": (
                    config_random_rollout_memory_learned_update(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_memory_learned_scale": (
                    config_random_rollout_memory_learned_scale(config) if config_random_rollout_memory(config) else None
                ),
                **(
                    {"random_rollout_memory_steps": config_random_rollout_memory_steps(config)}
                    if config_random_rollout_memory(config) and config_random_rollout_memory_steps(config) != 1
                    else {}
                ),
                **(
                    {
                        "random_rollout_memory_multiscale_steps": (
                            config_random_rollout_memory_multiscale_steps(config)
                        )
                    }
                    if config_random_rollout_memory(config)
                    and config_random_rollout_memory_multiscale_steps(config)
                    != [config_random_rollout_memory_steps(config)]
                    else {}
                ),
                **(
                    {
                        "random_rollout_memory_multiscale_weights": (
                            config_random_rollout_memory_multiscale_weights(config)
                        )
                    }
                    if config_random_rollout_memory(config)
                    and config_random_rollout_memory_multiscale_weights(config) is not None
                    else {}
                ),
                "random_rollout_head_merge": (
                    config_random_rollout_head_merge(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_weight_mode": (
                    config_random_rollout_weight_mode(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_edge_scope": (
                    config_random_rollout_edge_scope(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_local_logit_bias": (
                    config_random_rollout_local_logit_bias(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_cross_logit_bias": (
                    config_random_rollout_cross_logit_bias(config) if config_random_rollout_memory(config) else None
                ),
                "random_rollout_output_logits": config_random_rollout_output_logits(config),
                "random_rollout_output_scale": (
                    config_random_rollout_output_scale(config) if config_random_rollout_output_logits(config) else None
                ),
                "random_rollout_output_mode": (
                    config_random_rollout_output_mode(config) if config_random_rollout_output_logits(config) else None
                ),
                "random_attention_residual_scale": config_random_attention_residual_scale(config),
                "random_ffn_residual_scale": config_random_ffn_residual_scale(config),
                "random_history_output_logits": config_random_history_output_logits(config),
                "random_history_output_scale": (
                    config_random_history_output_scale(config) if config_random_history_output_logits(config) else None
                ),
                "random_history_output_source": (
                    config_random_history_output_source(config) if config_random_history_output_logits(config) else None
                ),
                "random_history_output_merge": (
                    config_random_history_output_merge(config) if config_random_history_output_logits(config) else None
                ),
                "random_history_include_input": (
                    config_random_history_include_input(config) if config_random_history_output_logits(config) else None
                ),
                "random_positional_rollout_memory": config_random_positional_rollout_memory(config),
                "random_positional_rollout_scale": (
                    config_random_positional_rollout_scale(config)
                    if config_random_positional_rollout_memory(config)
                    else None
                ),
                "random_positional_rollout_update": (
                    config_random_positional_rollout_update(config)
                    if config_random_positional_rollout_memory(config)
                    else None
                ),
                "random_positional_rollout_head_merge": (
                    config_random_positional_rollout_head_merge(config)
                    if config_random_positional_rollout_memory(config)
                    else None
                ),
                "random_positional_rollout_output_logits": config_random_positional_rollout_output_logits(config),
                "random_positional_rollout_output_scale": (
                    config_random_positional_rollout_output_scale(config)
                    if config_random_positional_rollout_output_logits(config)
                    else None
                ),
                "random_positional_rollout_output_mode": (
                    config_random_positional_rollout_output_mode(config)
                    if config_random_positional_rollout_output_logits(config)
                    else None
                ),
                "random_token_rollout_memory": config_random_token_rollout_memory(config),
                "random_token_rollout_scale": (
                    config_random_token_rollout_scale(config) if config_random_token_rollout_memory(config) else None
                ),
                "random_token_rollout_logit_mode": (
                    config_random_token_rollout_logit_mode(config) if config_random_token_rollout_memory(config) else None
                ),
            }
        )
    return identity


def identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _legacy_identity_without_added_defaults(identity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a narrow legacy identity variant for checkpoints predating new default fields.

    Older rollout checkpoints were written before these three fields were included in
    run_identity.  Their default values do not change the model forward pass or loaded
    weights, so final-eval can safely recognize that exact historical identity while
    still rejecting any other mismatch.
    """

    legacy_identity = dict(identity)
    removed_defaults: dict[str, Any] = {}
    added_default_fields = {
        "resume_model_strict": True,
        "random_rollout_memory_learned_update": False,
        "random_rollout_memory_learned_scale": False,
    }
    for key, default_value in added_default_fields.items():
        if legacy_identity.get(key) == default_value:
            removed_defaults[key] = legacy_identity.pop(key)
    return legacy_identity, removed_defaults


def checkpoint_identity_compatibility(
    payload: dict[str, Any],
    identity: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Validate checkpoint identity, with a narrow compatibility path for legacy defaults."""

    current_identity_sha = identity_sha256(identity)
    checkpoint_identity_sha = payload.get("identity_sha256")
    if checkpoint_identity_sha == current_identity_sha:
        return current_identity_sha, None

    legacy_identity, removed_defaults = _legacy_identity_without_added_defaults(identity)
    if removed_defaults:
        legacy_identity_sha = identity_sha256(legacy_identity)
        if checkpoint_identity_sha == legacy_identity_sha:
            return current_identity_sha, {
                "status": "legacy_added_default_fields_match",
                "checkpoint_identity_sha256": checkpoint_identity_sha,
                "current_identity_sha256": current_identity_sha,
                "legacy_identity_sha256": legacy_identity_sha,
                "removed_default_fields": removed_defaults,
                "reason": (
                    "checkpoint predates identity fields that default to the current model behavior; "
                    "all other identity fields match"
                ),
            }

    raise RuntimeError("checkpoint identity does not match current config/data/graph/code")


def artifact_args(record: dict[str, Any], method: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        block_size=int(record["resolved_graph_block_size"]),
        degree=int(record["resolved_graph_degree_or_budget"]),
        causal=False,
        graph_config=record["graph_artifacts"]["artifact"],
        graph_artifact=record["graph_artifacts"]["artifact"],
        graph_certificate=record["graph_artifacts"]["certificate"],
        graph_artifact_path=record["graph_artifacts"]["selected_graph_path"],
        seed=int(seed),
        random_alignment_mode="per_query_noncausal_unique_k",
        random_target_k_source="zigzag_actual_noncausal_per_query_unique_k",
        multiplicity_mode="unique_log_m",
    )


def layerwise_zigzag_graph_config(record: dict[str, Any], seed_base: int, layer_index: int) -> dict[str, Any]:
    original = record["graph_artifacts"]["artifact"]
    return build_graph_artifact(
        N_task=int(original.get("N_task", record["resolved_train_examples"])),
        T_raw=int(record["resolved_raw_sequence_length"]),
        block_size=int(record["resolved_graph_block_size"]),
        degree=int(record["resolved_graph_degree_or_budget"]),
        graph_seed=int(seed_base) + int(layer_index),
        g_config=original.get("G", {}),
        h_config=original.get("H", {}),
        version=str(original.get("version", "v06")),
    )


def method_artifact_args(
    record: dict[str, Any],
    method: str,
    seed: int,
    config: dict[str, Any] | None = None,
    layer_index: int | None = None,
) -> SimpleNamespace:
    args = artifact_args(record, method, seed)
    if layer_index is not None:
        args.random_base_seed = int(seed)
        args.random_layer_index = int(layer_index)
        args.seed = int(seed) + 104729 * (int(layer_index) + 1)
    if method in {"zigzag_certified", "zigzag_boolean"}:
        config = config or {}
        if config_layerwise_zigzag_random_graphs(config):
            layer = int(layer_index or 0)
            graph_seed_base = config_zigzag_graph_seed_base(config, seed)
            graph_seed = graph_seed_base + layer
            graph_config = layerwise_zigzag_graph_config(record, graph_seed_base, layer)
            args.graph_config = graph_config
            args.graph_artifact = graph_config
            args.graph_artifact_path = f"inline_layerwise_random_graph_seed{graph_seed}"
            args.zigzag_layer_index = layer
            args.zigzag_graph_seed = int(graph_seed)
            args.multiplicity_mode = "boolean" if method == "zigzag_boolean" else "unique_log_m"
    if method == "random_regular":
        config = config or {}
        density = config_random_density(config)
        args.random_include_local_edges = config_random_include_local_edges(config)
        if config_random_multihop_copy_route(config):
            if density is None:
                raise ValueError("random_multihop_copy_route requires random_actual_mask_density")
            if not config_random_include_local_edges(config):
                raise ValueError("random_multihop_copy_route requires random_include_local_edges=true")
            args.random_actual_mask_density = float(density)
            args.random_alignment_mode = "actual_mask_density_multihop_copy_route"
            args.random_target_k_source = "configured_actual_mask_density_multihop_copy_route"
            args.random_multihop_copy_route = True
            args.random_route_layers = int(record["resolved_layers"])
            args.random_route_source_length = int(record["resolved_runtime_target_length"])
            args.random_route_target_start = int(record["resolved_readout_start"])
            route_stride = config_random_route_stride(config)
            if route_stride is not None:
                args.random_route_stride = int(route_stride)
            args.random_route_multiplicity = config_random_route_multiplicity(config)
            args.random_use_log_m = config_random_route_use_log_m(config)
            args.random_route_layerwise_staged = config_random_route_layerwise_staged(config)
            args.random_route_transport = config_random_route_transport(config)
            args.random_route_transport_scale = config_random_route_transport_scale(config)
            args.random_route_transport_mode = config_random_route_transport_mode(config)
            args.random_aligned_rows = build_random_remote_rows_for_multihop_copy_route(
                int(record["resolved_padded_sequence_length"]),
                args,
                float(density),
            )
        elif density is None:
            if not config_random_include_local_edges(config):
                raise ValueError("random_include_local_edges=false requires random_actual_mask_density")
            args.random_aligned_rows = build_random_remote_rows_aligned_to_zigzag_noncausal(
                int(record["resolved_padded_sequence_length"]),
                args,
            )
        else:
            args.random_actual_mask_density = float(density)
            if config_random_include_local_edges(config):
                args.random_alignment_mode = "actual_mask_density"
                args.random_target_k_source = "configured_actual_mask_density"
                args.random_aligned_rows = build_random_remote_rows_for_actual_mask_density(
                    int(record["resolved_padded_sequence_length"]),
                    args,
                    float(density),
                )
            else:
                args.random_alignment_mode = "pure_actual_mask_density_no_local"
                args.random_target_k_source = "configured_pure_random_actual_mask_density"
                args.random_aligned_rows = build_pure_random_rows_for_actual_mask_density(
                    int(record["resolved_padded_sequence_length"]),
                    args,
                    float(density),
                    exclude_block_local=True,
                )
    return args


def aggregate_layerwise_artifacts(layer_artifacts: list[Any]) -> SimpleNamespace:
    first = layer_artifacts[0]
    metrics = dict(first.metrics)
    per_layer_metrics = [dict(item.metrics) for item in layer_artifacts]
    pair_counts = [int(item.metrics.get("attention_pair_count", 0)) for item in layer_artifacts]
    local_pair_counts = [int(item.metrics.get("local_attention_pair_count", 0)) for item in layer_artifacts]
    unique_k_means = [float(item.metrics.get("unique_k_mean", 0.0)) for item in layer_artifacts]
    multiplicity_maxes = [int(item.metrics.get("multiplicity_max", 0)) for item in layer_artifacts]
    multiplicity_means = [float(item.metrics.get("multiplicity_mean_nonzero", 0.0)) for item in layer_artifacts]
    graph_seeds = [item.metrics.get("zigzag_graph_seed") for item in layer_artifacts if "zigzag_graph_seed" in item.metrics]
    graph_ids = [item.metrics.get("zigzag_graph_id") for item in layer_artifacts if "zigzag_graph_id" in item.metrics]
    metrics.update(
        {
            "layerwise_independent_masks": True,
            "layerwise_mask_count": len(layer_artifacts),
            "per_layer_attention_pair_count": pair_counts,
            "per_layer_local_attention_pair_count": local_pair_counts,
            "per_layer_unique_k_mean": unique_k_means,
            "per_layer_multiplicity_max": multiplicity_maxes,
            "per_layer_multiplicity_mean_nonzero": multiplicity_means,
            "attention_pair_count_min_across_layers": min(pair_counts) if pair_counts else 0,
            "attention_pair_count_max_across_layers": max(pair_counts) if pair_counts else 0,
            "local_attention_pair_count_min_across_layers": min(local_pair_counts) if local_pair_counts else 0,
            "local_attention_pair_count_max_across_layers": max(local_pair_counts) if local_pair_counts else 0,
            "unique_k_mean_min_across_layers": min(unique_k_means) if unique_k_means else 0.0,
            "unique_k_mean_max_across_layers": max(unique_k_means) if unique_k_means else 0.0,
            "multiplicity_max_min_across_layers": min(multiplicity_maxes) if multiplicity_maxes else 0,
            "multiplicity_max_max_across_layers": max(multiplicity_maxes) if multiplicity_maxes else 0,
            "multiplicity_mean_min_across_layers": min(multiplicity_means) if multiplicity_means else 0.0,
            "multiplicity_mean_max_across_layers": max(multiplicity_means) if multiplicity_means else 0.0,
            "per_layer_zigzag_graph_seed": graph_seeds,
            "per_layer_zigzag_graph_id": graph_ids,
            "per_layer_metrics": per_layer_metrics,
        }
    )
    return SimpleNamespace(
        mask=[item.mask for item in layer_artifacts],
        local_valid=[item.local_valid for item in layer_artifacts],
        neighbors=[item.neighbors for item in layer_artifacts],
        valid_neighbors=[item.valid_neighbors for item in layer_artifacts],
        block_pair_index=[item.block_pair_index for item in layer_artifacts],
        local_log_m=[item.local_log_m for item in layer_artifacts],
        neighbor_log_m=[item.neighbor_log_m for item in layer_artifacts],
        route_transport_src=[getattr(item, "route_transport_src", None) for item in layer_artifacts],
        route_transport_dst=[getattr(item, "route_transport_dst", None) for item in layer_artifacts],
        route_transport_scale=[getattr(item, "route_transport_scale", None) for item in layer_artifacts],
        route_transport_mode=[getattr(item, "route_transport_mode", None) for item in layer_artifacts],
        metrics=metrics,
    )


def layer_value(value: Any, layer_index: int):
    if isinstance(value, (list, tuple)):
        return value[layer_index]
    return value


def edge_bias_shapes_from_artifacts(artifacts: Any, layers: int) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for layer_index in range(int(layers)):
        local_valid = layer_value(artifacts.local_valid, layer_index)
        neighbors = layer_value(artifacts.neighbors, layer_index)
        if local_valid is None:
            raise ValueError("learned edge parameters require local_valid artifacts")
        seq_len = int(local_valid.shape[0])
        local_width = int(local_valid.shape[1])
        neighbor_width = int(neighbors.shape[1]) if neighbors is not None else 0
        shapes.append((seq_len, local_width, neighbor_width))
    return shapes


def edge_bias_parameter_count_from_shapes(shapes: list[tuple[int, int, int]] | None) -> int:
    if not shapes:
        return 0
    return int(sum(int(seq_len) * (int(local_width) + int(neighbor_width)) for seq_len, local_width, neighbor_width in shapes))


def _mask_rows_as_sets(mask: Any) -> list[set[int]]:
    if isinstance(mask, torch.Tensor):
        cpu_mask = mask.detach().to(device="cpu", dtype=torch.bool)
        return [set(int(v) for v in torch.nonzero(cpu_mask[row], as_tuple=False).flatten().tolist()) for row in range(cpu_mask.shape[0])]
    raise TypeError(f"expected tensor mask, got {type(mask)!r}")


def copy_reachability_metrics(artifacts: Any, record: dict[str, Any]) -> dict[str, Any]:
    layers = int(record["resolved_layers"])
    seq_len = int(record["resolved_padded_sequence_length"])
    target_len = int(record["resolved_runtime_target_length"])
    target_start = int(record["resolved_readout_start"])
    masks = artifacts.mask if isinstance(artifacts.mask, (list, tuple)) else [artifacts.mask] * layers
    row_sets_by_layer = [_mask_rows_as_sets(layer_mask) for layer_mask in masks]
    reached_by_hop = {hop: 0 for hop in range(1, layers + 1)}
    shortest_values: list[int] = []
    direct_count = 0
    for offset in range(target_len):
        source = offset
        frontier = {target_start + offset}
        found = None
        for hop in range(1, layers + 1):
            next_frontier: set[int] = set()
            rows = row_sets_by_layer[hop - 1]
            for row in frontier:
                if 0 <= row < seq_len:
                    next_frontier.update(rows[row])
            frontier = next_frontier
            if hop == 1 and source in frontier:
                direct_count += 1
            if source in frontier:
                found = hop
                break
        if found is not None:
            shortest_values.append(found)
            for hop in range(found, layers + 1):
                reached_by_hop[hop] += 1
    hist = Counter(shortest_values)
    unreachable = target_len - len(shortest_values)
    out: dict[str, Any] = {
        **{f"copy_target_in_{hop}hop_rate": reached_by_hop[hop] / max(target_len, 1) for hop in range(1, layers + 1)},
        "copy_target_in_Lhop_rate": reached_by_hop[layers] / max(target_len, 1),
        "copy_average_shortest_path": (sum(shortest_values) / len(shortest_values)) if shortest_values else None,
        "copy_unreachable_rate": unreachable / max(target_len, 1),
        "copy_per_target_shortest_path_histogram": {str(key): int(value) for key, value in sorted(hist.items())},
        "copy_direct_source_edge_count": int(direct_count),
        "copy_direct_source_edge_rate": direct_count / max(target_len, 1),
        "copy_reachability_direction_definition": (
            "mask[marker_query_position, source_key_position] == true; "
            "multi-hop follows query-to-key rows backward through transformer layers"
        ),
        "copy_reachability_task_specific_structure": False,
    }
    return out


def build_model(
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    config: dict[str, Any] | None = None,
):
    backend = resolve_attention_backend(str(record["resolved_attention_backend"]), method)
    artifact_device = torch.device("cpu") if device.type == "cuda" else device
    layerwise_random = method == "random_regular" and config_random_layerwise_independent(config or {})
    layerwise_zigzag = method in {"zigzag_certified", "zigzag_boolean"} and config_layerwise_zigzag_random_graphs(config or {})
    if layerwise_random or layerwise_zigzag:
        layer_artifacts = []
        for layer_index in range(int(record["resolved_layers"])):
            layer_args = method_artifact_args(record, method, seed, config, layer_index=layer_index)
            layer_artifact = make_attention_artifacts(
                method,
                int(record["resolved_padded_sequence_length"]),
                layer_args,
                artifact_device,
                backend,
            )
            if layerwise_zigzag:
                layer_artifact.metrics["zigzag_layer_index"] = int(layer_index)
                layer_artifact.metrics["zigzag_graph_seed"] = int(layer_args.zigzag_graph_seed)
                layer_artifact.metrics["zigzag_graph_id"] = str(layer_args.graph_config.get("graph_id", ""))
            layer_artifacts.append(layer_artifact)
        artifacts = aggregate_layerwise_artifacts(layer_artifacts)
    else:
        args = method_artifact_args(record, method, seed, config)
        artifacts = make_attention_artifacts(method, int(record["resolved_padded_sequence_length"]), args, artifact_device, backend)
    artifacts = attention_artifacts_to_device(artifacts, device)
    config = config or {}
    learned_attention_edge_bias = method == "random_regular" and config_random_learned_attention_edge_bias(config)
    learned_edge_memory_transport_mode = (
        config_random_learned_edge_memory_transport_mode(config) if method == "random_regular" else None
    )
    edge_bias_shapes = None
    if learned_attention_edge_bias or learned_edge_memory_transport_mode is not None:
        if backend not in {"split", "blockpair"}:
            raise ValueError("learned random edge parameters require split or blockpair attention backend")
        edge_bias_shapes = edge_bias_shapes_from_artifacts(artifacts, int(record["resolved_layers"]))
        artifacts.metrics.update(
            {
                "random_learned_attention_edge_bias": bool(learned_attention_edge_bias),
                "random_learned_edge_memory_transport_mode": learned_edge_memory_transport_mode,
                "random_learned_edge_memory_transport_scale": config_random_learned_edge_memory_transport_scale(config),
                "random_learned_edge_memory_transport_temperature": config_random_learned_edge_memory_transport_temperature(config),
                "random_learned_edge_bias_init": config_random_learned_edge_bias_init(config),
                "random_learned_edge_bias_parameter_count": edge_bias_parameter_count_from_shapes(edge_bias_shapes),
                "random_learned_edge_bias_shapes": [list(shape) for shape in edge_bias_shapes],
                "random_learned_edge_task_specific_structure": False,
            }
        )
    elif method == "random_regular":
        artifacts.metrics.update(
            {
                "random_learned_attention_edge_bias": False,
                "random_learned_edge_memory_transport_mode": None,
                "random_learned_edge_bias_parameter_count": 0,
            }
        )
    value_position_encoding = config_random_value_position_encoding(config) if method == "random_regular" else "none"
    relative_attention_bias = method == "random_regular" and config_random_relative_attention_bias(config)
    attention_logit_scale_multiplier = (
        config_random_attention_logit_scale_multiplier(config) if method == "random_regular" else 1.0
    )
    learned_attention_logit_scale = method == "random_regular" and config_random_learned_attention_logit_scale(config)
    attention_top_k = config_random_attention_top_k(config) if method == "random_regular" else 0
    rollout_memory = method == "random_regular" and config_random_rollout_memory(config)
    rollout_output_logits = method == "random_regular" and config_random_rollout_output_logits(config)
    attention_residual_scale = config_random_attention_residual_scale(config) if method == "random_regular" else 1.0
    ffn_residual_scale = config_random_ffn_residual_scale(config) if method == "random_regular" else 1.0
    history_output_logits = method == "random_regular" and config_random_history_output_logits(config)
    positional_rollout_memory = method == "random_regular" and config_random_positional_rollout_memory(config)
    positional_rollout_output_logits = (
        method == "random_regular" and config_random_positional_rollout_output_logits(config)
    )
    token_rollout_memory = method == "random_regular" and config_random_token_rollout_memory(config)
    if method == "random_regular":
        artifacts.metrics.update(
            {
                "random_include_local_edges": config_random_include_local_edges(config),
                "random_pure_no_local_edges": not config_random_include_local_edges(config),
                "random_value_position_encoding": value_position_encoding,
                "random_value_position_encoding_task_specific_structure": False,
                "random_relative_attention_bias": bool(relative_attention_bias),
                "random_relative_attention_bias_parameter_count": (
                    int(record["resolved_layers"])
                    * int(record["resolved_heads"])
                    * (2 * int(record["resolved_padded_sequence_length"]) - 1)
                    if relative_attention_bias
                    else 0
                ),
                "random_relative_attention_bias_task_specific_structure": False,
                "random_attention_logit_scale_multiplier": attention_logit_scale_multiplier,
                "random_learned_attention_logit_scale": bool(learned_attention_logit_scale),
                "random_attention_logit_scale_task_specific_structure": False,
                "random_attention_top_k": int(attention_top_k),
                "random_attention_top_k_task_specific_structure": False,
                "random_rollout_memory": bool(rollout_memory),
                "random_rollout_memory_scale": config_random_rollout_memory_scale(config) if rollout_memory else None,
                "random_rollout_memory_source": config_random_rollout_memory_source(config) if rollout_memory else None,
                "random_rollout_memory_update": config_random_rollout_memory_update(config) if rollout_memory else None,
                "random_rollout_memory_lazy_alpha": (
                    config_random_rollout_memory_lazy_alpha(config)
                    if rollout_memory and config_random_rollout_memory_update(config) == "lazy"
                    else None
                ),
                "random_rollout_memory_learned_update": (
                    config_random_rollout_memory_learned_update(config) if rollout_memory else None
                ),
                "random_rollout_memory_learned_scale": (
                    config_random_rollout_memory_learned_scale(config) if rollout_memory else None
                ),
                "random_rollout_memory_steps": config_random_rollout_memory_steps(config) if rollout_memory else None,
                "random_rollout_memory_multiscale_steps": (
                    config_random_rollout_memory_multiscale_steps(config) if rollout_memory else None
                ),
                "random_rollout_memory_multiscale_weights": (
                    config_random_rollout_memory_multiscale_weights(config) if rollout_memory else None
                ),
                "random_rollout_head_merge": config_random_rollout_head_merge(config) if rollout_memory else None,
                "random_rollout_weight_mode": config_random_rollout_weight_mode(config) if rollout_memory else None,
                "random_rollout_edge_scope": config_random_rollout_edge_scope(config) if rollout_memory else None,
                "random_rollout_local_logit_bias": config_random_rollout_local_logit_bias(config) if rollout_memory else None,
                "random_rollout_cross_logit_bias": config_random_rollout_cross_logit_bias(config) if rollout_memory else None,
                "random_rollout_memory_task_specific_structure": False,
                "random_rollout_output_logits": bool(rollout_output_logits),
                "random_rollout_output_scale": (
                    config_random_rollout_output_scale(config) if rollout_output_logits else None
                ),
                "random_rollout_output_mode": (
                    config_random_rollout_output_mode(config) if rollout_output_logits else None
                ),
                "random_rollout_output_task_specific_structure": False,
                "random_attention_residual_scale": attention_residual_scale,
                "random_ffn_residual_scale": ffn_residual_scale,
                "random_residual_scale_task_specific_structure": False,
                "random_history_output_logits": bool(history_output_logits),
                "random_history_output_scale": (
                    config_random_history_output_scale(config) if history_output_logits else None
                ),
                "random_history_output_source": (
                    config_random_history_output_source(config) if history_output_logits else None
                ),
                "random_history_output_merge": (
                    config_random_history_output_merge(config) if history_output_logits else None
                ),
                "random_history_include_input": (
                    config_random_history_include_input(config) if history_output_logits else None
                ),
                "random_history_output_task_specific_structure": False,
                "random_positional_rollout_memory": bool(positional_rollout_memory),
                "random_positional_rollout_scale": (
                    config_random_positional_rollout_scale(config) if positional_rollout_memory else None
                ),
                "random_positional_rollout_update": (
                    config_random_positional_rollout_update(config) if positional_rollout_memory else None
                ),
                "random_positional_rollout_head_merge": (
                    config_random_positional_rollout_head_merge(config) if positional_rollout_memory else None
                ),
                "random_positional_rollout_output_logits": bool(positional_rollout_output_logits),
                "random_positional_rollout_output_scale": (
                    config_random_positional_rollout_output_scale(config)
                    if positional_rollout_output_logits
                    else None
                ),
                "random_positional_rollout_output_mode": (
                    config_random_positional_rollout_output_mode(config)
                    if positional_rollout_output_logits
                    else None
                ),
                "random_positional_rollout_task_specific_structure": False,
                "random_token_rollout_memory": bool(token_rollout_memory),
                "random_token_rollout_scale": config_random_token_rollout_scale(config) if token_rollout_memory else None,
                "random_token_rollout_logit_mode": (
                    config_random_token_rollout_logit_mode(config) if token_rollout_memory else None
                ),
                "random_token_rollout_task_specific_structure": False,
            }
        )
    artifacts.metrics.update(copy_reachability_metrics(artifacts, record))
    seed_policy = set_all_seeds(int(record.get("model_seed", seed)))
    model = ProbeTransformer(
        vocab_size=int(record["resolved_vocab_or_value_space_size"]),
        token_output_size=int(record["resolved_token_output_size"]),
        class_count=2,
        seq_len=int(record["resolved_padded_sequence_length"]),
        d_model=int(record["resolved_d_model"]),
        layers=int(record["resolved_layers"]),
        heads=int(record["resolved_heads"]),
        ffn_dim=int(record["resolved_ffn_dim"]),
        dropout=float(record["resolved_dropout"]),
        attention_backend=backend,
        block_size=int(record["resolved_graph_block_size"]),
        position_encoding=str(record["position_encoding"]),
        rope_theta=float(record["rope_theta"]),
        use_class_head=False,
        edge_bias_shapes=edge_bias_shapes,
        learned_attention_edge_bias=learned_attention_edge_bias,
        learned_edge_memory_transport_mode=learned_edge_memory_transport_mode,
        learned_edge_memory_transport_scale=config_random_learned_edge_memory_transport_scale(config),
        learned_edge_memory_transport_temperature=config_random_learned_edge_memory_transport_temperature(config),
        learned_edge_bias_init=config_random_learned_edge_bias_init(config),
        value_position_encoding=value_position_encoding,
        relative_attention_bias=relative_attention_bias,
        relative_attention_bias_init=config_random_relative_attention_bias_init(config),
        attention_logit_scale_multiplier=attention_logit_scale_multiplier,
        learned_attention_logit_scale=learned_attention_logit_scale,
        attention_top_k=attention_top_k,
        rollout_memory=rollout_memory,
        rollout_memory_scale=config_random_rollout_memory_scale(config),
        rollout_memory_source=config_random_rollout_memory_source(config),
        rollout_memory_update=config_random_rollout_memory_update(config),
        rollout_memory_lazy_alpha=config_random_rollout_memory_lazy_alpha(config),
        rollout_memory_learned_update=config_random_rollout_memory_learned_update(config),
        rollout_memory_learned_scale=config_random_rollout_memory_learned_scale(config),
        rollout_memory_steps=config_random_rollout_memory_steps(config),
        rollout_memory_multiscale_steps=config_random_rollout_memory_multiscale_steps(config),
        rollout_memory_multiscale_weights=config_random_rollout_memory_multiscale_weights(config),
        rollout_head_merge=config_random_rollout_head_merge(config),
        rollout_weight_mode=config_random_rollout_weight_mode(config),
        rollout_edge_scope=config_random_rollout_edge_scope(config),
        rollout_local_logit_bias=config_random_rollout_local_logit_bias(config),
        rollout_cross_logit_bias=config_random_rollout_cross_logit_bias(config),
        rollout_output_logits=rollout_output_logits,
        rollout_output_scale=config_random_rollout_output_scale(config),
        rollout_output_mode=config_random_rollout_output_mode(config),
        attention_residual_scale=attention_residual_scale,
        ffn_residual_scale=ffn_residual_scale,
        history_output_logits=history_output_logits,
        history_output_scale=config_random_history_output_scale(config),
        history_output_source=config_random_history_output_source(config),
        history_output_merge=config_random_history_output_merge(config),
        history_include_input=config_random_history_include_input(config),
        positional_rollout_memory=positional_rollout_memory,
        positional_rollout_scale=config_random_positional_rollout_scale(config),
        positional_rollout_update=config_random_positional_rollout_update(config),
        positional_rollout_head_merge=config_random_positional_rollout_head_merge(config),
        positional_rollout_output_logits=positional_rollout_output_logits,
        positional_rollout_output_scale=config_random_positional_rollout_output_scale(config),
        positional_rollout_output_mode=config_random_positional_rollout_output_mode(config),
        token_rollout_memory=token_rollout_memory,
        token_rollout_scale=config_random_token_rollout_scale(config),
        token_rollout_logit_mode=config_random_token_rollout_logit_mode(config),
    ).to(device)
    return model, artifacts, backend, seed_policy


def gate_model_record(record: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    mapping = {
        "layers": "resolved_layers",
        "d_model": "resolved_d_model",
        "heads": "resolved_heads",
        "ffn_dim": "resolved_ffn_dim",
        "dropout": "resolved_dropout",
        "batch_size": "resolved_batch_size",
        "gradient_accumulation_steps": "resolved_gradient_accumulation_steps",
        "learning_rate": "resolved_base_learning_rate",
    }
    for cfg_key, record_key in mapping.items():
        if cfg_key in gate_cfg:
            out[record_key] = gate_cfg[cfg_key]
    out["resolved_effective_batch_size"] = int(out["resolved_batch_size"]) * int(out["resolved_gradient_accumulation_steps"])
    return out


def train_model_record(record: dict[str, Any], train_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    mapping = {
        "learning_rate": "resolved_base_learning_rate",
        "base_learning_rate": "resolved_base_learning_rate",
        "min_learning_rate": "resolved_min_learning_rate",
        "lr_scheduler": "resolved_lr_scheduler",
        "weight_decay": "resolved_weight_decay",
        "grad_clip_norm": "resolved_grad_clip_norm",
        "batch_size": "resolved_batch_size",
        "gradient_accumulation_steps": "resolved_gradient_accumulation_steps",
        "layers": "resolved_layers",
        "d_model": "resolved_d_model",
        "heads": "resolved_heads",
        "ffn_dim": "resolved_ffn_dim",
        "dropout": "resolved_dropout",
    }
    for cfg_key, record_key in mapping.items():
        if cfg_key in train_cfg:
            out[record_key] = train_cfg[cfg_key]
    out["resolved_effective_batch_size"] = int(out["resolved_batch_size"]) * int(out["resolved_gradient_accumulation_steps"])
    return out


def evaluate_rows(
    model,
    artifacts,
    encoder,
    record: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(raw_rows), batch_size):
            batch_rows = raw_rows[start_idx : start_idx + batch_size]
            batch = make_probe_batch(batch_rows, record, encoder, device)
            _loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record, config)
            rows.extend(per_sample)
    elapsed = max(time.perf_counter() - start, 1e-9)
    agg = aggregate_metric_rows(rows, record["primary_metric"])
    if rows:
        margin_min_values = [
            float(row["copy_token_margin_min"])
            for row in rows
            if isinstance(row.get("copy_token_margin_min"), (int, float))
            and math.isfinite(float(row["copy_token_margin_min"]))
        ]
        margin_mean_numer = sum(
            float(row.get("copy_token_margin_mean", 0.0)) * max(int(row.get("tokens", 0)), 1)
            for row in rows
            if isinstance(row.get("copy_token_margin_mean"), (int, float))
            and math.isfinite(float(row["copy_token_margin_mean"]))
        )
        margin_mean_denom = sum(
            max(int(row.get("tokens", 0)), 1)
            for row in rows
            if isinstance(row.get("copy_token_margin_mean"), (int, float))
            and math.isfinite(float(row["copy_token_margin_mean"]))
        )
        if margin_min_values:
            agg["secondary_metrics"]["copy_token_margin_min"] = min(margin_min_values)
        if margin_mean_denom:
            agg["secondary_metrics"]["copy_token_margin_mean"] = margin_mean_numer / max(margin_mean_denom, 1)
    agg["elapsed_sec"] = elapsed
    agg["examples_per_sec"] = agg["examples"] / elapsed
    agg["tokens_per_sec"] = agg["tokens"] / elapsed
    model.train()
    return agg


def _selected_copy_logits(model, artifacts, batch) -> torch.Tensor:
    token_logits, _class_logits = model(
        batch.tokens,
        batch.pad_mask,
        artifacts.mask,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        artifacts.block_pair_index,
        artifacts.local_log_m,
        artifacts.neighbor_log_m,
        getattr(artifacts, "route_transport_src", None),
        getattr(artifacts, "route_transport_dst", None),
        getattr(artifacts, "route_transport_scale", None),
        getattr(artifacts, "route_transport_mode", None),
    )
    if batch.target_positions is None:
        raise ValueError("copy margin requires target_positions")
    expected = torch.arange(1024, 2048, device=batch.target_positions.device).repeat(batch.target_positions.shape[0], 1)
    if not bool(torch.equal(batch.target_positions, expected)):
        raise ValueError("copy_corrected_v01 target_positions must be exactly 1024..2047")
    return token_logits[:, 1024:2048, :]


def _token_margins(selected: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    true_logits = selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    wrong_logits = selected.masked_fill(
        F.one_hot(targets, num_classes=selected.shape[-1]).to(dtype=torch.bool, device=selected.device),
        -torch.inf,
    )
    runner_up = wrong_logits.max(dim=-1).values
    return (true_logits - runner_up)[target_mask]


def copy_margin_row_stats(selected: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    margins = selected.new_empty((0,))
    for index in range(selected.shape[0]):
        mask = target_mask[index]
        sample_margins = _token_margins(selected[index : index + 1], targets[index : index + 1], mask[None, :])
        rows.append(
            {
                "copy_token_margin_min": float(sample_margins.min().item()) if sample_margins.numel() else math.nan,
                "copy_token_margin_mean": float(sample_margins.mean().item()) if sample_margins.numel() else math.nan,
                "copy_token_margin_p01": float(torch.quantile(sample_margins.float(), 0.01).item())
                if sample_margins.numel()
                else math.nan,
            }
        )
        margins = torch.cat([margins, sample_margins.detach()])
    return rows


def forward_loss_and_metrics(model, artifacts, batch, task_record: dict, config: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict, list[dict]]:
    if not bool(task_record.get("copy_corrected_v01", False)):
        raise ValueError("copy_corrected runner only supports copy_corrected_v01")
    assert batch.targets is not None and batch.target_mask is not None
    selected = _selected_copy_logits(model, artifacts, batch)
    loss_sum = masked_sequence_loss_sum(selected, batch.targets, batch.target_mask)
    token_count = int(batch.target_mask.sum().item())
    ce_loss = loss_sum / max(token_count, 1)
    margins = _token_margins(selected, batch.targets, batch.target_mask)
    margin_target = config_copy_margin_target(config or {})
    margin_loss = F.relu(float(margin_target) - margins).mean() if margins.numel() else ce_loss.new_zeros(())
    margin_weight = config_copy_margin_loss_weight(config or {})
    loss = ce_loss + float(margin_weight) * margin_loss
    metrics = sequence_metrics(selected, batch.targets, batch.target_mask)
    pred = selected.argmax(dim=-1)
    margin_stats = copy_margin_row_stats(selected, batch.targets, batch.target_mask)
    per_sample = []
    for index, subtask in enumerate(batch.subtasks):
        mask = batch.target_mask[index]
        token_total = int(mask.sum().item())
        token_correct = int(((pred[index] == batch.targets[index]) & mask).sum().item())
        exact = bool((((pred[index] == batch.targets[index]) | ~mask).all()).item())
        sample_loss_sum = F.cross_entropy(selected[index][mask], batch.targets[index][mask], reduction="sum")
        row = {
            "examples": 1,
            "tokens": token_total,
            "loss": float(sample_loss_sum.item()) / max(token_total, 1),
            "loss_sum": float(sample_loss_sum.item()),
            "subtask": subtask,
            "token_accuracy": token_correct / max(token_total, 1),
            "exact_match": 1.0 if exact else 0.0,
            "copy_token_accuracy": token_correct / max(token_total, 1),
            "copy_sequence_accuracy": 1.0 if exact else 0.0,
            "copy_margin_loss": float(margin_loss.item()),
            "copy_margin_loss_weight": float(margin_weight),
            "copy_margin_target": float(margin_target),
            **margin_stats[index],
        }
        per_sample.append(row)
    metrics.update(
        {
            "copy_token_accuracy": metrics["token_accuracy"],
            "copy_sequence_accuracy": metrics["sequence_accuracy"],
            "copy_token_margin_min": float(margins.min().item()) if margins.numel() else math.nan,
            "copy_token_margin_mean": float(margins.mean().item()) if margins.numel() else math.nan,
            "copy_token_margin_p01": float(torch.quantile(margins.float(), 0.01).item()) if margins.numel() else math.nan,
            "copy_margin_loss": float(margin_loss.item()),
            "copy_margin_loss_weight": float(margin_weight),
            "copy_margin_target": float(margin_target),
        }
    )
    return loss, metrics, per_sample


def make_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    record: dict[str, Any],
    identity: dict[str, Any],
    epoch: int,
    optimizer_step: int,
    micro_step: int,
    permutation_position: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {"type": record["resolved_lr_scheduler"]},
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "sampler": {
            "data_seed": int(record.get("data_seed", 0)),
            "epoch": int(epoch),
            "permutation_position": int(permutation_position),
            "policy": "without_replacement_full_permutation",
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "identity": identity,
        "identity_sha256": identity_sha256(identity),
        "created_at": utc_now(),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": tensor_checkpoint_sha256(path),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "permutation_position": int(permutation_position),
    }


def train_loop(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, method, seed, mode=mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config_path, config, manifest_path, record, method, seed)
    identity_hash = identity_sha256(identity)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        existing = read_json(summary_path)
        if existing.get("status") == "ok" and existing.get("identity_sha256") == identity_hash and existing.get("mode") == mode:
            return existing
        if existing.get("status") == "ok":
            raise RuntimeError("stale successful summary exists with different run identity; refusing to skip")
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    command = command_string()
    write_command(run_dir / "command.sh", command)
    write_json(run_dir / "raw_config_snapshot.json", config)

    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_cfg = dict(config.get("train", {}))
    overfit = dict(config.get("gate_overfit", {}))
    active_record = gate_model_record(record, overfit) if mode == "gate-overfit" else train_model_record(record, train_cfg)
    write_json(run_dir / "resolved_config_snapshot.json", active_record)
    write_json(run_dir / "run_identity.json", identity)
    model, artifacts, backend, seed_policy = build_model(active_record, method, seed, device, config)
    initial_hash = state_dict_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(active_record["resolved_base_learning_rate"]),
        weight_decay=float(active_record["resolved_weight_decay"]),
    )
    resume_checkpoint = config_resume_from_checkpoint(config)
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_payload = load_torch_checkpoint(resume_path, device)
        resume_model_strict = config_resume_model_strict(config)
        load_result = model.load_state_dict(resume_payload["model_state"], strict=resume_model_strict)
        if not resume_model_strict:
            write_json(
                run_dir / "non_strict_resume_model_load.json",
                {
                    "resume_from_checkpoint": str(resume_path),
                    "missing_keys": list(load_result.missing_keys),
                    "unexpected_keys": list(load_result.unexpected_keys),
                    "reason": "explicit resume_model_strict=false; newly added generic random-rollout parameters start from config initialization",
                },
            )
        if (
            "optimizer_state" in resume_payload
            and bool(dict(config.get("train", {})).get("resume_optimizer", False))
            and resume_model_strict
        ):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if bool(dict(config.get("train", {})).get("resume_rng", True)):
            rng_state = dict(resume_payload.get("rng_state", {}))
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch_cpu" in rng_state:
                torch.set_rng_state(rng_state["torch_cpu"].detach().cpu())
            if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all([state.detach().cpu() for state in rng_state["torch_cuda"]])
        initial_hash = state_dict_sha256(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if mode == "gate-overfit":
        max_steps = int(overfit.get("max_steps", 300))
        fixed_rows = [train_store.row(i) for i in range(int(overfit.get("examples", 2)))]
        batch_size = len(fixed_rows)
        accum = 1
        base_lr = float(overfit.get("learning_rate", active_record["resolved_base_learning_rate"]))
        checkpoint_every = 0
        epochs = 0
    else:
        max_steps = int(math.ceil(len(train_store) / int(active_record["resolved_effective_batch_size"]))) * int(train_cfg.get("epochs", 1))
        if train_cfg.get("max_steps") is not None:
            max_steps = int(train_cfg["max_steps"])
        batch_size = int(active_record["resolved_batch_size"])
        accum = int(active_record["resolved_gradient_accumulation_steps"])
        base_lr = float(active_record["resolved_base_learning_rate"])
        checkpoint_every = int(train_cfg.get("checkpoint_every", 100))
        epochs = int(train_cfg.get("epochs", 1))

    metrics_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    train_loss_last = math.nan
    train_loss_sum = 0.0
    train_token_count = 0
    log_every = int(train_cfg.get("log_every", 25))
    diagnostic_rows = [train_store.row(i) for i in range(int(train_cfg.get("train_diagnostic_examples", 16)))]
    global_step = 0
    resume_start_step = int(resume_payload.get("optimizer_step", 0)) if resume_payload is not None else 0
    started = time.perf_counter()
    prev_log = started
    model.train()

    if mode == "gate-overfit":
        for step in range(1, max_steps + 1):
            global_step = step
            for group in optimizer.param_groups:
                group["lr"] = base_lr
            optimizer.zero_grad(set_to_none=True)
            batch = make_probe_batch(fixed_rows, record, encoder, device)
            loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at gate-overfit step {step}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(active_record["resolved_grad_clip_norm"])))
            optimizer.step()
            train_loss_last = float(loss.item())
            train_loss_sum += sum(float(item["loss_sum"]) for item in per_sample)
            train_token_count += sum(int(item["tokens"]) for item in per_sample)
            diag = evaluate_rows(model, artifacts, encoder, record, fixed_rows, batch_size, device, config)
            now = time.perf_counter()
            row = {
                "run_id": run_id_for(config, record, method, seed),
                "mode": mode,
                "task": "copy",
                "method": method,
                "seed": int(seed),
                "step": step,
                "split": "train_overfit",
                "phase": "gate2_single_batch_overfit",
                "timestamp_utc": utc_now(),
                "train_loss": train_loss_last,
                "epoch_mean_loss": train_loss_sum / max(train_token_count, 1),
                "primary_metric_value": diag["primary_metric_value"],
                "secondary_metrics_json": json_metric(diag["secondary_metrics"]),
                "learning_rate": base_lr,
                "grad_norm": grad_norm,
                "gate_model": {
                    "layers": active_record["resolved_layers"],
                    "d_model": active_record["resolved_d_model"],
                    "heads": active_record["resolved_heads"],
                    "ffn_dim": active_record["resolved_ffn_dim"],
                },
                "elapsed_sec_total": now - started,
                "seconds_since_prev_log": now - prev_log,
                "tokens_per_sec": diag["tokens_per_sec"],
                "examples_per_sec": diag["examples_per_sec"],
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
            }
            metrics_rows.append(row)
            prev_log = now
            print(json.dumps(row, ensure_ascii=False), flush=True)
            seq_acc = diag["secondary_metrics"].get("copy_sequence_accuracy", 0.0)
            if (
                diag["primary_metric_value"] >= float(config["gate_overfit"]["threshold_token_accuracy"])
                and seq_acc >= float(config["gate_overfit"]["threshold_sequence_accuracy"])
                and diag["loss"] <= float(config["gate_overfit"]["threshold_loss"])
            ):
                break
    else:
        effective_batch = batch_size * accum
        for epoch in range(epochs):
            permutation = deterministic_permutation(len(train_store), int(record.get("data_seed", seed)), epoch)
            coverage = epoch_coverage(permutation, len(train_store))
            coverage["epoch"] = epoch
            coverage_rows.append(coverage)
            position = 0
            while position < len(permutation):
                global_step += 1
                absolute_step = resume_start_step + global_step
                lr = schedule_lr(
                    str(active_record["resolved_lr_scheduler"]),
                    base_lr,
                    float(active_record["resolved_min_learning_rate"]),
                    0,
                    max_steps,
                    global_step,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum = 0.0
                step_tokens = 0
                for micro in range(accum):
                    batch_indices = permutation[position : position + batch_size]
                    if not batch_indices:
                        break
                    if len(batch_indices) != batch_size:
                        raise RuntimeError("copy_corrected_v01 expects full micro-batches; adjust batch/accum to divide train rows")
                    position += batch_size
                    raw_rows = [train_store.row(index) for index in batch_indices]
                    batch = make_probe_batch(raw_rows, record, encoder, device)
                    loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record, config)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at step={global_step}")
                    (loss / accum).backward()
                    step_loss_sum += sum(float(item["loss_sum"]) for item in per_sample)
                    step_tokens += sum(int(item["tokens"]) for item in per_sample)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(active_record["resolved_grad_clip_norm"])))
                optimizer.step()
                train_loss_last = step_loss_sum / max(step_tokens, 1)
                train_loss_sum += step_loss_sum
                train_token_count += step_tokens
                if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
                    diag = evaluate_rows(
                        model,
                        artifacts,
                        encoder,
                        record,
                        diagnostic_rows,
                        int(record["resolved_eval_batch_size"]),
                        device,
                        config,
                    )
                    now = time.perf_counter()
                    row = {
                        "run_id": run_id_for(config, record, method, seed),
                        "mode": mode,
                        "task": "copy",
                        "method": method,
                        "seed": int(seed),
                        "step": global_step,
                        "absolute_step": absolute_step,
                        "resume_start_step": resume_start_step,
                        "epoch": epoch,
                        "split": "train_diagnostic",
                        "phase": "train_no_test_read",
                        "timestamp_utc": utc_now(),
                        "train_loss_last_step": train_loss_last,
                        "train_loss_epoch_mean_so_far": train_loss_sum / max(train_token_count, 1),
                        "eval_loss": diag["loss"],
                        "primary_metric_value": diag["primary_metric_value"],
                        "secondary_metrics_json": json_metric(diag["secondary_metrics"]),
                        "learning_rate": lr,
                        "grad_norm": grad_norm,
                        "elapsed_sec_total": now - started,
                        "seconds_since_prev_log": now - prev_log,
                        "tokens_per_sec": diag["tokens_per_sec"],
                        "examples_per_sec": diag["examples_per_sec"],
                        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
                    }
                    metrics_rows.append(row)
                    prev_log = now
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    append_jsonl(metrics_path, row)
                if checkpoint_every and global_step % checkpoint_every == 0:
                    checkpoints.append(
                        make_checkpoint(
                            run_dir / "checkpoints" / f"checkpoint_step{global_step}.pt",
                            model,
                            optimizer,
                            active_record,
                            identity,
                            epoch,
                            global_step,
                            0,
                            position,
                        )
                    )
                if global_step >= max_steps:
                    break
    final_checkpoint = make_checkpoint(
        run_dir / "checkpoints" / f"{mode}_final_step{global_step}.pt",
        model,
        optimizer,
        active_record,
        identity,
        epochs - 1 if epochs else 0,
        global_step,
        0,
        0,
    )
    checkpoints.append(final_checkpoint)
    write_checkpoint_manifest(run_dir, checkpoints, "tensor_checkpoint_with_optimizer_sampler_rng_state")
    if not metrics_path.exists():
        write_jsonl(metrics_path, metrics_rows)
    if coverage_rows:
        write_json(run_dir / "sampler_coverage.json", {"epochs": coverage_rows})
    summary = {
        "status": "ok",
        "mode": mode,
        "version": experiment_version(config, record),
        "run_id": run_id_for(config, record, method, seed),
        "method": method,
        "seed": int(seed),
        "identity": identity,
        "identity_sha256": identity_hash,
        "command": command,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "backend": backend,
        "attention_metrics": artifacts.metrics,
        "actual_mask_density": (
            float(artifacts.metrics.get("attention_pair_count", 0))
            / float(int(active_record["resolved_padded_sequence_length"]) ** 2)
        ),
        "seed_policy": seed_policy,
        "active_model": {
            "layers": active_record["resolved_layers"],
            "d_model": active_record["resolved_d_model"],
            "heads": active_record["resolved_heads"],
            "ffn_dim": active_record["resolved_ffn_dim"],
            "dropout": active_record["resolved_dropout"],
        },
        "initial_model_state_sha256": initial_hash,
        "resume_from_checkpoint": resume_checkpoint,
        "resume_model_strict": config_resume_model_strict(config),
        "resume_start_step": resume_start_step,
        "parameter_count": parameter_count(model),
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "train_loss_last_step": train_loss_last,
        "train_loss_mean": train_loss_sum / max(train_token_count, 1),
        "steps_completed": global_step,
        "checkpoint": final_checkpoint,
        "metrics_path": str(metrics_path),
        "checkpoint_manifest_path": str(run_dir / "checkpoint_manifest.json"),
        "test_read_during_training": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    if metrics_rows:
        last = metrics_rows[-1]
        summary["last_primary_metric_value"] = last.get("primary_metric_value")
        try:
            summary["last_secondary_metrics"] = json.loads(last.get("secondary_metrics_json", "{}"))
        except Exception:
            summary["last_secondary_metrics"] = {}
    write_json(summary_path, summary)
    return summary


def train_baselines(train_store: JsonlStore) -> dict[str, Any]:
    global_counts = np.zeros(64, dtype=np.int64)
    position_counts = np.zeros((1024, 64), dtype=np.int64)
    total = 0
    for rows in train_store.batches(32):
        for row in rows:
            target = row["target"]
            for pos, token in enumerate(target):
                global_counts[int(token)] += 1
                position_counts[pos, int(token)] += 1
                total += 1
    probs = global_counts / max(total, 1)
    nonzero = probs > 0
    empirical_nll = -sum(float(global_counts[i]) * math.log(float(probs[i])) for i in range(64) if nonzero[i]) / max(total, 1)
    return {
        "uniform64_accuracy": 1.0 / 64.0,
        "uniform64_nll": math.log(64.0),
        "target_support_min": 1,
        "target_support_max": 62,
        "target_support_size": 62,
        "global_mode_token": int(global_counts.argmax()),
        "global_mode_token_accuracy": float(global_counts.max() / max(total, 1)),
        "empirical_train_marginal_nll": empirical_nll,
        "position_wise_mode_token_accuracy": float(position_counts.max(axis=1).sum() / max(total, 1)),
    }


def final_eval(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    checkpoint: Path | None,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, method, seed, mode="train")
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        ckpt_manifest = read_json(run_dir / "checkpoint_manifest.json")
        checkpoint = Path(ckpt_manifest["latest_checkpoint"]["path"])
    identity = run_identity(config_path, config, manifest_path, record, method, seed)
    model, artifacts, backend, seed_policy = build_model(record, method, seed, device, config)
    payload = load_torch_checkpoint(checkpoint, device)
    identity_hash, identity_compatibility = checkpoint_identity_compatibility(payload, identity)
    model.load_state_dict(payload["model_state"])
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    baselines = train_baselines(train_store)
    first_test_read_at = utc_now()
    test_store = JsonlStore(Path(record["version_path"]) / "test.jsonl")
    test_rows = [test_store.row(i) for i in range(len(test_store))]
    result = evaluate_rows(model, artifacts, encoder, record, test_rows, int(record["resolved_eval_batch_size"]), device, config)
    out = {
        "status": "ok",
        "mode": "final-eval",
        "version": experiment_version(config, record),
        "run_id": run_id_for(config, record, method, seed, "final_eval"),
        "method": method,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": tensor_checkpoint_sha256(checkpoint),
        "checkpoint_identity_sha256": payload.get("identity_sha256"),
        "identity": identity,
        "identity_sha256": identity_hash,
        "identity_compatibility": identity_compatibility,
        "backend": backend,
        "attention_metrics": artifacts.metrics,
        "actual_mask_density": (
            float(artifacts.metrics.get("attention_pair_count", 0))
            / float(int(record["resolved_padded_sequence_length"]) ** 2)
        ),
        "seed_policy": seed_policy,
        "first_test_read_at": first_test_read_at,
        "test_examples": result["examples"],
        "test_target_tokens": result["tokens"],
        "test_loss": result["loss"],
        "copy_token_accuracy": result["primary_metric_value"],
        "copy_sequence_accuracy": result["secondary_metrics"].get("copy_sequence_accuracy", 0.0),
        "copy_token_margin_min": result["secondary_metrics"].get("copy_token_margin_min", math.nan),
        "copy_token_margin_mean": result["secondary_metrics"].get("copy_token_margin_mean", math.nan),
        "copy_token_margin_p01": result["secondary_metrics"].get("copy_token_margin_p01", math.nan),
        "copy_margin_loss": result["secondary_metrics"].get("copy_margin_loss", math.nan),
        "secondary_metrics": result["secondary_metrics"],
        "baselines": baselines,
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": record["resolved_train_split_sha256"],
        "test_sha256": record["resolved_test_split_sha256"],
        "graph_sha256": record["graph_artifacts"]["selected_graph_sha256"],
        "target_positions": "1024..2047",
        "T": 2048,
        "padding_positions": 0,
    }
    write_json(run_dir / "final_eval.json", out)
    fieldnames = sorted(k for k, value in out.items() if not isinstance(value, (dict, list)))
    with (run_dir / "final_eval.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: out.get(key) for key in fieldnames})
    return out


def train_eval(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    examples: int | None,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, method, seed, mode="train")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config_path, config, manifest_path, record, method, seed)
    model, artifacts, backend, seed_policy = build_model(record, method, seed, device, config)
    payload = load_torch_checkpoint(checkpoint, device)
    identity_hash, identity_compatibility = checkpoint_identity_compatibility(payload, identity)
    model.load_state_dict(payload["model_state"])
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    limit = len(train_store) if examples is None or int(examples) <= 0 else min(int(examples), len(train_store))
    train_rows = [train_store.row(i) for i in range(limit)]
    result = evaluate_rows(model, artifacts, encoder, record, train_rows, int(record["resolved_eval_batch_size"]), device, config)
    out = {
        "status": "ok",
        "mode": "train-eval",
        "version": experiment_version(config, record),
        "run_id": run_id_for(config, record, method, seed, f"train_eval_{limit}"),
        "method": method,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": tensor_checkpoint_sha256(checkpoint),
        "checkpoint_identity_sha256": payload.get("identity_sha256"),
        "identity": identity,
        "identity_sha256": identity_hash,
        "identity_compatibility": identity_compatibility,
        "backend": backend,
        "attention_metrics": artifacts.metrics,
        "actual_mask_density": (
            float(artifacts.metrics.get("attention_pair_count", 0))
            / float(int(record["resolved_padded_sequence_length"]) ** 2)
        ),
        "seed_policy": seed_policy,
        "train_eval_examples": result["examples"],
        "train_eval_target_tokens": result["tokens"],
        "train_eval_loss": result["loss"],
        "copy_token_accuracy": result["primary_metric_value"],
        "copy_sequence_accuracy": result["secondary_metrics"].get("copy_sequence_accuracy", 0.0),
        "copy_token_margin_min": result["secondary_metrics"].get("copy_token_margin_min", math.nan),
        "copy_token_margin_mean": result["secondary_metrics"].get("copy_token_margin_mean", math.nan),
        "copy_token_margin_p01": result["secondary_metrics"].get("copy_token_margin_p01", math.nan),
        "copy_margin_loss": result["secondary_metrics"].get("copy_margin_loss", math.nan),
        "secondary_metrics": result["secondary_metrics"],
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": record["resolved_train_split_sha256"],
        "test_sha256": record["resolved_test_split_sha256"],
        "graph_sha256": record["graph_artifacts"]["selected_graph_sha256"],
        "target_positions": "1024..2047",
        "T": 2048,
        "padding_positions": 0,
        "test_read": False,
    }
    suffix = "full" if limit == len(train_store) else str(limit)
    out_path = run_dir / f"train_eval_{suffix}.json"
    write_json(out_path, out)
    out["train_eval_path"] = str(out_path)
    return out


def train_analyze(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    examples: int | None,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, method, seed, mode="train")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config_path, config, manifest_path, record, method, seed)
    model, artifacts, backend, seed_policy = build_model(record, method, seed, device, config)
    payload = load_torch_checkpoint(checkpoint, device)
    identity_hash, identity_compatibility = checkpoint_identity_compatibility(payload, identity)
    model.load_state_dict(payload["model_state"])
    model.eval()
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    limit = len(train_store) if examples is None or int(examples) <= 0 else min(int(examples), len(train_store))
    batch_size = int(record["resolved_eval_batch_size"])
    error_rows: list[dict[str, Any]] = []
    lowest_margin_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start_idx in range(0, limit, batch_size):
            batch_indices = list(range(start_idx, min(start_idx + batch_size, limit)))
            raw_rows = [train_store.row(i) for i in batch_indices]
            batch = make_probe_batch(raw_rows, record, encoder, device)
            selected = _selected_copy_logits(model, artifacts, batch)
            assert batch.targets is not None and batch.target_mask is not None
            pred = selected.argmax(dim=-1)
            true_logits = selected.gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
            wrong_logits = selected.masked_fill(
                F.one_hot(batch.targets, num_classes=selected.shape[-1]).to(dtype=torch.bool, device=selected.device),
                -torch.inf,
            )
            runner_up = wrong_logits.max(dim=-1).values
            margins = true_logits - runner_up
            for local_index, row_index in enumerate(batch_indices):
                mask = batch.target_mask[local_index]
                sample_margins = margins[local_index][mask]
                min_margin, min_offset_tensor = sample_margins.min(dim=0)
                min_offset = int(min_offset_tensor.item())
                sample_pred = pred[local_index][mask]
                sample_targets = batch.targets[local_index][mask]
                wrong_positions = torch.nonzero(sample_pred != sample_targets, as_tuple=False).flatten()
                meta = raw_rows[local_index].get("metadata") if isinstance(raw_rows[local_index].get("metadata"), dict) else {}
                row_summary = {
                    "train_index": int(row_index),
                    "id": raw_rows[local_index].get("id"),
                    "variant": raw_rows[local_index].get("variant"),
                    "seed": meta.get("seed"),
                    "min_margin": float(min_margin.item()),
                    "min_offset": int(min_offset),
                    "min_position": int(1024 + min_offset),
                    "min_true": int(sample_targets[min_offset].item()),
                    "min_pred": int(sample_pred[min_offset].item()),
                    "wrong_count": int(wrong_positions.numel()),
                    "wrong_offsets": [int(v) for v in wrong_positions[:32].detach().cpu().tolist()],
                    "wrong_true": [int(sample_targets[v].item()) for v in wrong_positions[:32]],
                    "wrong_pred": [int(sample_pred[v].item()) for v in wrong_positions[:32]],
                }
                if row_summary["wrong_count"] > 0:
                    error_rows.append(row_summary)
                lowest_margin_rows.append(row_summary)
    lowest_margin_rows = sorted(lowest_margin_rows, key=lambda item: float(item["min_margin"]))[:32]
    out = {
        "status": "ok",
        "mode": "train-analyze",
        "version": experiment_version(config, record),
        "run_id": run_id_for(config, record, method, seed, f"train_analyze_{limit}"),
        "method": method,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": tensor_checkpoint_sha256(checkpoint),
        "checkpoint_identity_sha256": payload.get("identity_sha256"),
        "identity_sha256": identity_hash,
        "identity_compatibility": identity_compatibility,
        "backend": backend,
        "seed_policy": seed_policy,
        "train_examples_analyzed": int(limit),
        "error_sequence_count": len(error_rows),
        "error_rows": error_rows[:64],
        "lowest_margin_rows": lowest_margin_rows,
        "test_read": False,
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": record["resolved_train_split_sha256"],
        "test_sha256": record["resolved_test_split_sha256"],
        "target_positions": "1024..2047",
        "T": 2048,
    }
    out_path = run_dir / ("train_analyze_full.json" if limit == len(train_store) else f"train_analyze_{limit}.json")
    write_json(out_path, out)
    out["train_analyze_path"] = str(out_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/copy_corrected_v01.json"))
    parser.add_argument("--mode", choices=["gate-overfit", "train", "final-eval", "train-eval", "train-analyze"], required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-eval-examples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["task_parameter_manifest"])
    manifest = load_manifest(config)
    record = task_record(manifest)
    device = select_device(args.device)
    methods = [args.method] if args.method else list(config.get("methods", ["dense"]))
    seeds = [args.seed] if args.seed is not None else list(config.get("seeds", [0]))
    if args.mode == "gate-overfit":
        methods = [str(config.get("gate_overfit", {}).get("method", "dense"))] if args.method is None else methods
    outputs = []
    for method in methods:
        for seed in seeds:
            if args.mode in {"gate-overfit", "train"}:
                outputs.append(
                    train_loop(
                        config_path=args.config,
                        config=config,
                        manifest_path=manifest_path,
                        record=record,
                        method=method,
                        seed=int(seed),
                        device=device,
                        mode=args.mode,
                    )
                )
            elif args.mode == "final-eval":
                outputs.append(
                    final_eval(
                        config_path=args.config,
                        config=config,
                        manifest_path=manifest_path,
                        record=record,
                        method=method,
                        seed=int(seed),
                        device=device,
                        checkpoint=args.checkpoint,
                    )
                )
            elif args.mode == "train-eval":
                if args.checkpoint is None:
                    raise ValueError("--checkpoint is required for train-eval")
                outputs.append(
                    train_eval(
                        config_path=args.config,
                        config=config,
                        manifest_path=manifest_path,
                        record=record,
                        method=method,
                        seed=int(seed),
                        device=device,
                        checkpoint=args.checkpoint,
                        examples=args.train_eval_examples,
                    )
                )
            else:
                if args.checkpoint is None:
                    raise ValueError("--checkpoint is required for train-analyze")
                outputs.append(
                    train_analyze(
                        config_path=args.config,
                        config=config,
                        manifest_path=manifest_path,
                        record=record,
                        method=method,
                        seed=int(seed),
                        device=device,
                        checkpoint=args.checkpoint,
                        examples=args.train_eval_examples,
                    )
                )
    print(json.dumps({"status": "ok", "mode": args.mode, "runs": outputs}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
