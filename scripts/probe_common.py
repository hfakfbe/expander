from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_VERSION = "v08"
ATTENTION_CONTRACT = "non_causal"
GRAPH_DIRECTIONALITY = "directed"

SELECTED_PROBES: dict[str, dict[str, Any]] = {
    "copy": {
        "version_path": "../expander_bench/data/probes/copy/s4_copying_length_extrapolation/copy_s4_l0_m1024_a64_full_v2",
        "primary_metric": "copy_token_accuracy",
        "loss_family": "sequence_generation",
    },
    "selective_copy": {
        "version_path": "../expander_bench/data/probes/selective_copy/s4_variable_copy_regenerated/selective_copy_s4_l4096_m16_a16_full_v1",
        "primary_metric": "selective_copy_token_accuracy",
        "loss_family": "sequence_generation",
    },
    "induction_associative_recall": {
        "version_path": "../expander_bench/data/probes/induction_associative_recall/zoology_mqar_regenerated/mqar_vocab8192_len64_128_256_512_1024_full_v2",
        "primary_metric": "retrieval_exact_match",
        "loss_family": "key_value_or_token_retrieval",
    },
    "niah_kv_retrieval": {
        "version_path": "../expander_bench/data/probes/niah_kv_retrieval/ruler_niah_single_1/niah_ruler_noise_4k_full_v2",
        "primary_metric": "retrieval_exact_match",
        "loss_family": "key_value_or_token_retrieval",
    },
    "ruler": {
        "version_path": "../expander_bench/data/probes/ruler/ruler_official_nonqa_synthetic_suite/ruler_nonqa_suite_4k_full_v2",
        "primary_metric": "retrieval_exact_match",
        "loss_family": "key_value_or_token_retrieval",
    },
    "lra_listops": {
        "version_path": "../expander_bench/data/probes/lra_listops/lra_official_generator_regenerated/lra_listops_regenerated_len500_2000_96k_2k_2k_full_v1",
        "primary_metric": "listops_accuracy",
        "loss_family": "classification",
    },
}

REQUIRED_VERSION_FILES = [
    "README.md",
    "deployment_report.md",
    "dataset_card.json",
    "config.yaml",
    "source.lock",
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "checksums.sha256",
    "deployment_status.yaml",
]

FIELD_CONTRACT: list[str] = [
    "version",
    "experiment_version",
    "phase",
    "task",
    "subtask",
    "variant",
    "run_id",
    "status",
    "failure_reason",
    "timestamp_utc",
    "host",
    "remote_host",
    "local_or_remote",
    "command",
    "command_sha256",
    "log_path",
    "error_log",
    "CUDA_VISIBLE_DEVICES",
    "gpu_id",
    "gpu_name",
    "python_version",
    "torch_version",
    "cuda_version",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "phase4_manifest_path",
    "phase4_manifest_sha256",
    "phase4_task_parameter_record_path",
    "phase4_task_parameter_record_sha256",
    "artifact_dir",
    "external_artifact_manifest_path",
    "version_path",
    "dataset",
    "dataset_source",
    "dataset_revision_or_hash",
    "dataset_cache_or_local_path",
    "dataset_card_path",
    "dataset_card_sha256",
    "deployment_status_path",
    "deployment_status_sha256",
    "source_lock_path",
    "source_lock_sha256",
    "checksums_path",
    "checksums_sha256",
    "train_path",
    "train_sha256",
    "validation_path",
    "validation_sha256",
    "test_path",
    "test_sha256",
    "train_examples",
    "validation_examples",
    "test_examples",
    "train_examples_used",
    "validation_examples_used",
    "test_examples_used",
    "train_split_policy",
    "validation_split_policy",
    "test_split_policy",
    "input_schema",
    "target_schema",
    "metadata_keys",
    "encoder_or_tokenizer",
    "data_readiness_path",
    "data_readiness_sha256",
    "tokenization_summary_path",
    "tokenization_summary_sha256",
    "wikitext_data_phase_dir",
    "tokenized_train_path",
    "tokenized_train_sha256",
    "tokenized_test_path",
    "tokenized_test_sha256",
    "tokenizer_or_encoder_path",
    "tokenizer_or_encoder_sha256",
    "tokenizer",
    "tokenizer_algorithm",
    "tokenizer_train_split",
    "tokenizer_path",
    "tokenizer_sha256",
    "tokenizer_min_frequency",
    "tokenizer_special_tokens",
    "pad_token",
    "eos_token",
    "unk_token",
    "vocab_size",
    "train_nonempty_rows",
    "test_nonempty_rows",
    "train_token_count",
    "train_block_count",
    "test_token_count",
    "test_block_count",
    "label_or_value_space",
    "loss_type",
    "primary_metric_name",
    "secondary_metric_names",
    "sequence_length_min",
    "sequence_length_mean",
    "sequence_length_p95",
    "sequence_length_max",
    "target_length_min",
    "target_length_mean",
    "target_length_p95",
    "target_length_max",
    "method",
    "method_role",
    "required_or_optional_method",
    "seed",
    "model_family",
    "layers",
    "d_model",
    "heads",
    "ffn_dim",
    "dropout",
    "parameter_count",
    "optimizer",
    "steps",
    "steps_planned",
    "steps_completed",
    "train_epochs",
    "train_epochs_planned",
    "train_epochs_completed",
    "train_steps",
    "train_budget_policy",
    "train_budget_unit",
    "train_budget_value",
    "completed_train_units",
    "train_examples_seen",
    "train_tokens_seen",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "eval_batch_size",
    "validation_eval_budget",
    "test_eval_budget",
    "eval_batches",
    "learning_rate",
    "base_learning_rate",
    "lr_scheduler",
    "warmup_ratio",
    "warmup_steps",
    "min_lr_ratio",
    "min_learning_rate",
    "cosine_total_steps",
    "weight_decay",
    "grad_clip_norm",
    "log_every",
    "log_step_policy",
    "logging_reference_train_steps",
    "min_logged_train_step_count",
    "planned_logged_train_step_count",
    "actual_logged_train_step_count",
    "log_coverage_ratio",
    "log_policy_satisfied",
    "eval_every",
    "checkpoint_every",
    "checkpoint_policy",
    "checkpoint_manifest_path",
    "attention_contract",
    "causal",
    "graph_directionality",
    "attention_backend",
    "graph_id",
    "graph_seed",
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
    "selected_graph_sha256",
    "graph_artifact_sha256",
    "graph_artifact_sha256_matches_canonical",
    "graph_certificate_sha256",
    "graph_block_policy",
    "graph_degree_or_budget_policy",
    "graph_block_size",
    "graph_num_blocks_or_nodes",
    "graph_degree",
    "N_total",
    "B",
    "q",
    "d",
    "N_total_v07_alias",
    "B_v07_alias",
    "q_v07_alias",
    "d_v07_alias",
    "v07_alias_replacement_reason",
    "G_type",
    "H_type",
    "allow_multiedges",
    "multiplicity_mode",
    "lambda_G",
    "mu_H",
    "rho_zigzag_bound",
    "rho_zigzag_certified",
    "rho_zigzag_exact",
    "rot_g_is_bijection",
    "P_G_row_stochastic_error",
    "P_G_col_stochastic_error",
    "P_H_row_stochastic_error",
    "P_H_col_stochastic_error",
    "graph_certified",
    "implementation_certified",
    "theory_aligned_method",
    "duplicate_rate",
    "self_loop_rate",
    "remote_local_overlap_mean",
    "collision_count_mean",
    "zigzag_actual_k_min_after_causal",
    "zigzag_actual_k_mean_after_causal",
    "zigzag_actual_k_max_after_causal",
    "zigzag_attention_pair_count_after_causal",
    "zigzag_actual_k_min_noncausal",
    "zigzag_actual_k_mean_noncausal",
    "zigzag_actual_k_max_noncausal",
    "zigzag_attention_pair_count_noncausal",
    "random_target_k_source",
    "random_actual_k_min_after_causal",
    "random_actual_k_mean_after_causal",
    "random_actual_k_max_after_causal",
    "random_attention_pair_count_after_causal",
    "random_actual_k_min_noncausal",
    "random_actual_k_mean_noncausal",
    "random_actual_k_max_noncausal",
    "random_attention_pair_count_noncausal",
    "random_k_alignment_error_mean",
    "random_k_alignment_error_max",
    "random_alignment_mode",
    "random_k_aligned_to_zigzag",
    "attention_pair_count_after_causal",
    "attention_k_min",
    "attention_k_mean",
    "attention_k_max",
    "attention_pair_count",
    "attention_diagnostics_path",
    "zigzag_budget_path",
    "random_budget_path",
    "train_loss_final",
    "validation_loss_final",
    "test_loss",
    "primary_metric_value",
    "secondary_metrics_json",
    "task_metrics_json",
    "final_train_loss",
    "test_perplexity",
    "validation_perplexity_if_applicable",
    "test_perplexity_if_applicable",
    "train_tokens_per_sec",
    "validation_tokens_per_sec",
    "test_tokens_per_sec",
    "train_examples_per_sec",
    "validation_examples_per_sec",
    "test_examples_per_sec",
    "total_wall_time_sec",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "data_prep_wall_time_sec",
    "seconds_since_prev_log_mean",
    "seconds_since_prev_log_max",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "oom_fallback_applied",
    "oom_fallback_reason",
    "training_curves_path",
    "metrics_path",
    "summary_path",
    "raw_config_snapshot_path",
    "resolved_config_snapshot_path",
    "result_field_audit_path",
    "copy_token_accuracy",
    "copy_sequence_accuracy",
    "copy_eos_accuracy",
    "copy_source_length",
    "eval_token_accuracy",
    "eval_sequence_accuracy",
    "eval_eos_accuracy",
    "target_in_1hop_rate",
    "target_in_2hop_rate",
    "target_in_Lhop_rate",
    "average_shortest_path",
    "unreachable_rate",
    "selective_copy_token_accuracy",
    "selective_copy_sequence_accuracy",
    "retrieval_exact_match",
    "retrieval_token_accuracy",
    "retrieval_answer_format",
    "ruler_subtask",
    "ruler_subtask_exact_match",
    "ruler_subtask_token_accuracy",
    "listops_accuracy",
    "listops_macro_accuracy",
    "listops_class_count",
]

METRICS_FIELDS = [
    "run_id",
    "task",
    "subtask",
    "method",
    "seed",
    "timestamp_utc",
    "step",
    "epoch",
    "elapsed_sec_total",
    "seconds_since_prev_log",
    "split",
    "phase",
    "train_loss",
    "running_train_loss",
    "eval_loss",
    "running_eval_loss",
    "primary_metric_name",
    "primary_metric_value",
    "secondary_metrics_json",
    "learning_rate",
    "lr_scheduler",
    "grad_norm",
    "tokens_per_sec",
    "examples_per_sec",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "nonfinite_loss_detected",
    "nan_detected",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (repo_root() / value).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def string_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def command_string() -> str:
    return shlex.join([sys.executable, *sys.argv])


def write_command(path: Path, command: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((command or command_string()) + "\n", encoding="utf-8")
    path.chmod(0o755)


def deployed_git_commit(cwd: Path | None = None) -> str | None:
    start = Path(cwd or repo_root()).resolve()
    if start.is_file():
        start = start.parent
    for parent in [start, *start.parents]:
        marker = parent / f".deployed_git_commit_{EXPERIMENT_VERSION}"
        if marker.exists():
            value = marker.read_text(encoding="utf-8").strip()
            return value or None
    return None


def git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return deployed_git_commit(cwd) or "unknown"


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(cwd or repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return deployed_git_commit(cwd) is None


def host_name() -> str:
    return socket.gethostname()


def percentile(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "p95": float(percentile(values, 0.95)),
        "max": float(max(values)),
    }


def selected_probe_path(task: str) -> Path:
    return resolve_repo_path(SELECTED_PROBES[task]["version_path"])


def ensure_no_forbidden_probe(task: str) -> None:
    if task in {"lra_pathfinder", "lra_pathx"}:
        raise ValueError(f"{task} is forbidden in v08 main evaluation")


def default_result_row() -> dict[str, Any]:
    return {field: "not_applicable" for field in FIELD_CONTRACT}


def audit_result_row(row: dict[str, Any], metrics_rows: list[dict], phase4_record: dict, resolved: dict) -> dict:
    missing = [field for field in FIELD_CONTRACT if field not in row]
    empty = [
        field
        for field in FIELD_CONTRACT
        if field in row and row[field] in {"", None, []}
    ]
    forbidden_hits = []
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    for spelling in ["noncaual", "non-casual"]:
        if spelling in payload:
            forbidden_hits.append(spelling)
    train_steps = int(row.get("logging_reference_train_steps") or 0)
    actual_logged = sum(1 for item in metrics_rows if item.get("split") == "train")
    min_logged = train_steps if train_steps < 100 else int((train_steps + 99) // 100)
    final_logged = any(int(item.get("step", -1)) == train_steps and item.get("split") == "train" for item in metrics_rows)
    log_ok = actual_logged >= min_logged and final_logged
    phase4_trace_missing = [
        key
        for key in [
            "task",
            "attention_contract",
            "causal",
            "graph_directionality",
            "resolved_train_budget_value",
            "resolved_log_every",
        ]
        if key not in phase4_record
    ]
    resolved_trace_missing = [
        key
        for key in ["task", "method", "attention_contract", "causal", "graph_directionality"]
        if key not in resolved
    ]
    manual_only = []
    for key, value in phase4_record.items():
        if key.endswith("_policy") and key.replace("_policy", "_value") not in phase4_record:
            continue
    status = (
        "passed"
        if not missing
        and not empty
        and not forbidden_hits
        and not phase4_trace_missing
        and not resolved_trace_missing
        and not manual_only
        and log_ok
        else "failed"
    )
    return {
        "expected_field_count": len(FIELD_CONTRACT),
        "present_field_count": len([field for field in FIELD_CONTRACT if field in row]),
        "missing_fields": missing,
        "empty_fields": empty,
        "not_applicable_fields": [field for field, value in row.items() if value == "not_applicable"],
        "not_applicable_reasons": row.get("task_metrics_json", "not_applicable"),
        "phase4_trace_missing_fields": phase4_trace_missing,
        "resolved_config_trace_missing_fields": resolved_trace_missing,
        "forbidden_spelling_hits": forbidden_hits,
        "manual_only_policy_fields_without_resolved_value": manual_only,
        "logging_reference_train_steps": train_steps,
        "min_logged_train_step_count": min_logged,
        "actual_logged_train_step_count": actual_logged,
        "log_coverage_ratio": actual_logged / max(train_steps, 1),
        "log_policy_satisfied": log_ok,
        "final_train_step_logged": final_logged,
        "status": status,
    }
