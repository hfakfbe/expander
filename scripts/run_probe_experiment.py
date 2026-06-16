from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from probe_common import (
    ATTENTION_CONTRACT,
    EXPERIMENT_VERSION,
    FIELD_CONTRACT,
    GRAPH_DIRECTIONALITY,
    audit_result_row,
    command_string,
    default_result_row,
    file_sha256,
    git_commit,
    git_dirty,
    host_name,
    read_json,
    string_sha256,
    utc_now,
    write_command,
    write_csv,
    write_json,
    write_jsonl,
)
from probe_metrics import (
    aggregate_metric_rows,
    classification_metrics,
    json_metric,
    masked_sequence_loss,
    sequence_metrics,
    write_training_curves,
)
from probe_tasks import (
    JsonlStore,
    ProbeTransformer,
    gather_position_logits,
    load_encoder,
    make_probe_batch,
    parameter_count,
)
from synthetic_mvp_core.artifacts import make_attention_artifacts, resolve_attention_backend


def load_config(path: Path) -> dict:
    return read_json(path)


def load_manifest(config: dict) -> dict:
    return read_json(Path(config["task_parameter_manifest"]))


def task_records(manifest: dict) -> dict[str, dict]:
    return {row["task"]: row for row in manifest["tasks"]}


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def schedule_lr(base_lr: float, min_lr: float, warmup_steps: int, total_steps: int, step: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_profile(task_record: dict, profile: str) -> dict:
    if profile not in {"smoke", "main"}:
        raise ValueError(f"unknown profile={profile}")
    profile_cfg = dict(task_record[profile])
    profile_cfg["profile"] = profile
    return profile_cfg


def stores(task_record: dict) -> dict[str, JsonlStore]:
    version_path = Path(task_record["version_path"])
    return {
        "train": JsonlStore(version_path / "train.jsonl"),
        "validation": JsonlStore(version_path / "validation.jsonl"),
        "test": JsonlStore(version_path / "test.jsonl"),
    }


def artifact_args(task_record: dict, method: str, seed: int) -> SimpleNamespace:
    graph = task_record["graph_artifacts"]["artifact"]
    cert = task_record["graph_artifacts"]["certificate"]
    return SimpleNamespace(
        block_size=int(task_record["resolved_graph_block_size"]),
        degree=int(task_record["resolved_graph_degree_or_budget"]),
        causal=False,
        graph_config=graph,
        graph_artifact=graph,
        graph_certificate=cert,
        graph_artifact_path=task_record["graph_artifacts"]["selected_graph_path"],
        seed=int(seed),
        random_alignment_mode="noncausal_sparse_budget",
        random_target_k_source="zigzag_actual_noncausal",
        multiplicity_mode="unique_log_m",
    )


def build_model_and_artifacts(task_record: dict, method: str, seed: int, device: torch.device):
    encoder = load_encoder(Path(task_record["resolved_tokenizer_or_encoder_path"]))
    backend = resolve_attention_backend(str(task_record["resolved_attention_backend"]), method)
    args = artifact_args(task_record, method, seed)
    artifacts = make_attention_artifacts(method, int(task_record["resolved_padded_sequence_length"]), args, device, backend)
    class_count = 10 if task_record["task"] == "lra_listops" else max(2, int(task_record["resolved_vocab_or_value_space_size"]))
    model = ProbeTransformer(
        vocab_size=int(task_record["resolved_vocab_or_value_space_size"]),
        token_output_size=int(task_record["resolved_vocab_or_value_space_size"]),
        class_count=class_count,
        seq_len=int(task_record["resolved_padded_sequence_length"]),
        d_model=int(task_record["resolved_d_model"]),
        layers=int(task_record["resolved_layers"]),
        heads=int(task_record["resolved_heads"]),
        ffn_dim=int(task_record["resolved_ffn_dim"]),
        dropout=float(task_record["resolved_dropout"]),
        attention_backend=backend,
        block_size=int(task_record["resolved_graph_block_size"]),
    ).to(device)
    return encoder, model, artifacts, backend


def forward_loss_and_metrics(model, artifacts, batch, task_record: dict) -> tuple[torch.Tensor, dict, list[dict]]:
    token_logits, class_logits = model(
        batch.tokens,
        batch.pad_mask,
        artifacts.mask,
        artifacts.local_valid,
        artifacts.neighbors,
        artifacts.valid_neighbors,
        artifacts.block_pair_index,
        artifacts.local_log_m,
        artifacts.neighbor_log_m,
    )
    loss_type = task_record["resolved_loss_type"]
    task = task_record["task"]
    per_sample = []
    if loss_type in {"sequence_cross_entropy", "retrieval_sequence_cross_entropy", "mqar_position_cross_entropy"}:
        assert batch.target_positions is not None and batch.targets is not None and batch.target_mask is not None
        selected = gather_position_logits(token_logits, batch.target_positions)
        loss = masked_sequence_loss(selected, batch.targets, batch.target_mask)
        metrics = sequence_metrics(selected, batch.targets, batch.target_mask)
        pred = selected.argmax(dim=-1)
        for index, subtask in enumerate(batch.subtasks):
            mask = batch.target_mask[index]
            token_total = int(mask.sum().item())
            token_correct = int(((pred[index] == batch.targets[index]) & mask).sum().item())
            exact = bool((((pred[index] == batch.targets[index]) | ~mask).all()).item())
            row = {
                "examples": 1,
                "tokens": token_total,
                "loss": float(loss.item()),
                "subtask": subtask,
                "token_accuracy": token_correct / max(token_total, 1),
                "exact_match": 1.0 if exact else 0.0,
            }
            if task == "copy":
                row["copy_token_accuracy"] = row["token_accuracy"]
                row["copy_sequence_accuracy"] = row["exact_match"]
            elif task == "selective_copy":
                row["selective_copy_token_accuracy"] = row["token_accuracy"]
                row["selective_copy_sequence_accuracy"] = row["exact_match"]
            else:
                row["retrieval_token_accuracy"] = row["token_accuracy"]
                row["retrieval_exact_match"] = row["exact_match"]
                if task == "ruler":
                    row["ruler_subtask_token_accuracy"] = row["token_accuracy"]
                    row["ruler_subtask_exact_match"] = row["exact_match"]
            per_sample.append(row)
        metrics.update(
            {
                "copy_token_accuracy": metrics["token_accuracy"] if task == "copy" else 0.0,
                "copy_sequence_accuracy": metrics["sequence_accuracy"] if task == "copy" else 0.0,
                "selective_copy_token_accuracy": metrics["token_accuracy"] if task == "selective_copy" else 0.0,
                "selective_copy_sequence_accuracy": metrics["sequence_accuracy"] if task == "selective_copy" else 0.0,
                "retrieval_token_accuracy": metrics["token_accuracy"] if task not in {"copy", "selective_copy"} else 0.0,
                "retrieval_exact_match": metrics["exact_match"] if task not in {"copy", "selective_copy"} else 0.0,
            }
        )
        return loss, metrics, per_sample
    if loss_type == "classification_cross_entropy":
        assert batch.class_targets is not None
        loss = F.cross_entropy(class_logits, batch.class_targets)
        metrics = classification_metrics(class_logits, batch.class_targets, 10)
        pred = class_logits.argmax(dim=-1)
        for index, subtask in enumerate(batch.subtasks):
            ok = bool((pred[index] == batch.class_targets[index]).item())
            per_sample.append(
                {
                    "examples": 1,
                    "tokens": 1,
                    "loss": float(loss.item()),
                    "subtask": subtask,
                    "accuracy": 1.0 if ok else 0.0,
                    "listops_accuracy": 1.0 if ok else 0.0,
                    "listops_macro_accuracy": 1.0 if ok else 0.0,
                }
            )
        metrics["listops_accuracy"] = metrics["accuracy"]
        metrics["listops_macro_accuracy"] = metrics["macro_accuracy"]
        return loss, metrics, per_sample
    raise ValueError(loss_type)


def evaluate(model, artifacts, encoder, task_record: dict, store: JsonlStore, limit: int, batch_size: int, device: torch.device) -> dict:
    model.eval()
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for raw_rows in store.batches(batch_size, limit=limit):
            batch = make_probe_batch(raw_rows, task_record, encoder, device)
            loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, task_record)
            for item in per_sample:
                item["loss"] = float(loss.item())
            rows.extend(per_sample)
    elapsed = max(time.perf_counter() - start, 1e-9)
    agg = aggregate_metric_rows(rows, task_record["primary_metric"])
    agg["elapsed_sec"] = elapsed
    agg["examples_per_sec"] = agg["examples"] / elapsed
    agg["tokens_per_sec"] = agg["tokens"] / elapsed
    model.train()
    return agg


def copy_graph_artifacts(task_record: dict, run_dir: Path) -> dict:
    graph_dir = run_dir / "artifacts" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    source = task_record["graph_artifacts"]
    mapping = {
        "selected_graph_path": graph_dir / "selected_graph.json",
        "graph_certificate_path": graph_dir / "graph_certificate.json",
        "graph_generation_path": graph_dir / "graph_generation.json",
    }
    for source_key, dst in mapping.items():
        shutil.copyfile(source[source_key], dst)
    shutil.copyfile(source["graph_artifact_sha256_path"], graph_dir / "graph_artifact.sha256")
    return {
        "graph_artifact_path": str(mapping["selected_graph_path"]),
        "graph_certificate_path": str(mapping["graph_certificate_path"]),
        "graph_generation_path": str(mapping["graph_generation_path"]),
        "graph_artifact_sha256": file_sha256(mapping["selected_graph_path"]),
        "graph_certificate_sha256": file_sha256(mapping["graph_certificate_path"]),
    }


def budget_payload(method: str, artifacts, task_record: dict, zigzag_artifacts=None) -> dict:
    metrics = artifacts.metrics
    payload = {
        "method": method,
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "actual_k_min_noncausal": metrics.get("effective_k_min", 0),
        "actual_k_mean_noncausal": metrics.get("effective_k_mean", 0.0),
        "actual_k_max_noncausal": metrics.get("effective_k_max", 0),
        "attention_pair_count_noncausal": metrics.get("attention_pair_count", 0),
        "actual_k_min_after_causal": "not_applicable",
        "actual_k_mean_after_causal": "not_applicable",
        "actual_k_max_after_causal": "not_applicable",
        "attention_pair_count_after_causal": "not_applicable",
        "raw_k": metrics.get("raw_k", "not_applicable"),
        "duplicate_rate": metrics.get("duplicate_rate", "not_applicable"),
        "self_loop_rate": metrics.get("self_loop_rate", "not_applicable"),
    }
    if method == "random_regular" and zigzag_artifacts is not None:
        z = zigzag_artifacts.metrics
        payload.update(
            {
                "random_target_k_source": "zigzag_actual_noncausal",
                "random_alignment_mode": "mean_noncausal_budget",
                "random_k_alignment_error_mean": abs(float(metrics.get("effective_k_mean", 0.0)) - float(z.get("effective_k_mean", 0.0))),
                "random_k_alignment_error_max": abs(int(metrics.get("effective_k_max", 0)) - int(z.get("effective_k_max", 0))),
                "random_k_aligned_to_zigzag": abs(float(metrics.get("effective_k_mean", 0.0)) - float(z.get("effective_k_mean", 0.0))) < 1.0,
            }
        )
    return payload


def metric_row(
    run_id: str,
    task_record: dict,
    method: str,
    seed: int,
    step: int,
    split: str,
    phase: str,
    train_loss: float,
    eval_result: dict,
    lr: float,
    elapsed: float,
    since_prev: float,
    grad_norm: float,
) -> dict:
    return {
        "run_id": run_id,
        "task": task_record["task"],
        "subtask": "all",
        "method": method,
        "seed": seed,
        "timestamp_utc": utc_now(),
        "step": step,
        "epoch": 0,
        "elapsed_sec_total": elapsed,
        "seconds_since_prev_log": since_prev,
        "split": split,
        "phase": phase,
        "train_loss": train_loss,
        "running_train_loss": train_loss,
        "eval_loss": eval_result.get("loss", "not_applicable"),
        "running_eval_loss": eval_result.get("loss", "not_applicable"),
        "primary_metric_name": task_record["primary_metric"],
        "primary_metric_value": eval_result.get("primary_metric_value", "not_applicable"),
        "secondary_metrics_json": json_metric(eval_result.get("secondary_metrics", {})),
        "learning_rate": lr,
        "lr_scheduler": task_record["resolved_lr_scheduler"],
        "grad_norm": grad_norm,
        "tokens_per_sec": eval_result.get("tokens_per_sec", 0.0),
        "examples_per_sec": eval_result.get("examples_per_sec", 0.0),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
        "nonfinite_loss_detected": False,
        "nan_detected": False,
    }


def build_result_row(
    task_record: dict,
    profile_cfg: dict,
    method: str,
    seed: int,
    phase: str,
    run_id: str,
    run_dir: Path,
    command: str,
    backend: str,
    param_count: int,
    graph_paths: dict,
    artifacts,
    zigzag_budget: dict,
    random_budget: dict,
    train_loss: float,
    validation_result: dict,
    test_result: dict,
    metrics_rows: list[dict],
    started: float,
) -> dict:
    row = default_result_row()
    version_dir = Path(task_record["version_path"])
    cert = task_record["graph_artifacts"]["certificate"]
    elapsed = max(time.perf_counter() - started, 1e-9)
    train_log_rows = [item for item in metrics_rows if item.get("split") == "train"]
    since_values = [float(item.get("seconds_since_prev_log", 0.0) or 0.0) for item in train_log_rows]
    row.update(
        {
            "version": EXPERIMENT_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "phase": phase,
            "task": task_record["task"],
            "subtask": "all",
            "variant": read_json(version_dir / "dataset_card.json").get("variant", "not_applicable"),
            "run_id": run_id,
            "status": "ok",
            "failure_reason": "not_applicable",
            "timestamp_utc": utc_now(),
            "host": host_name(),
            "remote_host": host_name(),
            "local_or_remote": "remote" if str(Path.cwd()).startswith("/home/huiwei/ysx") else "local",
            "command": command,
            "command_sha256": string_sha256(command),
            "log_path": os.environ.get("PROBE_V08_LOG_PATH", "not_applicable"),
            "error_log": "not_applicable",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not_applicable"),
            "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "not_applicable",
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "config_sha256": file_sha256(Path("configs/probes_v08_task_parameters.json")),
            "phase4_manifest_path": "configs/probes_v08_task_parameters.json",
            "phase4_manifest_sha256": file_sha256(Path("configs/probes_v08_task_parameters.json")),
            "phase4_task_parameter_record_path": str(run_dir / "phase4_task_parameter_record.json"),
            "phase4_task_parameter_record_sha256": file_sha256(run_dir / "phase4_task_parameter_record.json"),
            "artifact_dir": str(run_dir),
            "external_artifact_manifest_path": str(run_dir / "checkpoint_manifest.json"),
            "version_path": str(version_dir),
            "dataset": task_record["task"],
            "dataset_source": task_record["dataset_source"],
            "dataset_revision_or_hash": task_record["dataset_revision_or_hash"],
            "dataset_cache_or_local_path": str(version_dir),
            "dataset_card_path": str(version_dir / "dataset_card.json"),
            "dataset_card_sha256": task_record["dataset_card_sha256"],
            "deployment_status_path": str(version_dir / "deployment_status.yaml"),
            "deployment_status_sha256": task_record["deployment_status_sha256"],
            "source_lock_path": str(version_dir / "source.lock"),
            "source_lock_sha256": task_record["source_lock_sha256"],
            "checksums_path": str(version_dir / "checksums.sha256"),
            "checksums_sha256": task_record["checksums_sha256"],
            "train_path": str(version_dir / "train.jsonl"),
            "train_sha256": task_record["resolved_train_split_sha256"],
            "validation_path": str(version_dir / "validation.jsonl"),
            "validation_sha256": task_record["resolved_validation_split_sha256"],
            "test_path": str(version_dir / "test.jsonl"),
            "test_sha256": task_record["resolved_test_split_sha256"],
            "train_examples": task_record["resolved_train_examples"],
            "validation_examples": task_record["resolved_validation_examples"],
            "test_examples": task_record["resolved_test_examples"],
            "train_examples_used": min(int(task_record["resolved_train_examples"]), int(profile_cfg["steps"]) * int(task_record["resolved_effective_batch_size"])),
            "validation_examples_used": int(profile_cfg["validation_examples"]),
            "test_examples_used": int(profile_cfg["test_examples"]),
            "train_split_policy": task_record["train_split_policy"],
            "validation_split_policy": task_record["validation_split_policy"],
            "test_split_policy": task_record["test_split_policy"],
            "input_schema": task_record["input_schema"],
            "target_schema": task_record["target_schema"],
            "metadata_keys": "see_phase1_task_audit",
            "encoder_or_tokenizer": task_record["encoder_or_tokenizer"],
            "tokenizer_or_encoder_path": task_record["resolved_tokenizer_or_encoder_path"],
            "tokenizer_or_encoder_sha256": task_record["resolved_tokenizer_or_encoder_sha256"],
            "tokenizer": task_record["resolved_encoder_type"],
            "tokenizer_algorithm": task_record["resolved_encoder_type"],
            "tokenizer_train_split": "train",
            "tokenizer_path": task_record["resolved_tokenizer_or_encoder_path"],
            "tokenizer_sha256": task_record["resolved_tokenizer_or_encoder_sha256"],
            "tokenizer_min_frequency": "not_applicable",
            "tokenizer_special_tokens": "pad=0,eos=1,unk=2",
            "pad_token": 0,
            "eos_token": 1,
            "unk_token": 2,
            "vocab_size": task_record["resolved_vocab_or_value_space_size"],
            "train_nonempty_rows": task_record["resolved_train_examples"],
            "test_nonempty_rows": task_record["resolved_test_examples"],
            "label_or_value_space": task_record["label_or_value_space"],
            "loss_type": task_record["resolved_loss_type"],
            "primary_metric_name": task_record["primary_metric"],
            "secondary_metric_names": ",".join(task_record["secondary_metrics"]),
            "sequence_length_min": task_record["resolved_sequence_length_min"],
            "sequence_length_mean": task_record["resolved_sequence_length_mean"],
            "sequence_length_p95": task_record["resolved_sequence_length_p95"],
            "sequence_length_max": task_record["resolved_sequence_length_max"],
            "target_length_min": task_record["resolved_target_length_min"],
            "target_length_mean": task_record["resolved_target_length_mean"],
            "target_length_p95": task_record["resolved_target_length_p95"],
            "target_length_max": task_record["resolved_target_length_max"],
            "method": method,
            "method_role": "main_theory_aligned" if method == "zigzag_certified" else "baseline",
            "required_or_optional_method": "required" if method in task_record["resolved_required_methods"] else "optional",
            "seed": seed,
            "model_family": task_record["resolved_model_family"],
            "layers": task_record["resolved_layers"],
            "d_model": task_record["resolved_d_model"],
            "heads": task_record["resolved_heads"],
            "ffn_dim": task_record["resolved_ffn_dim"],
            "dropout": task_record["resolved_dropout"],
            "parameter_count": param_count,
            "optimizer": task_record["resolved_optimizer"],
            "steps": int(profile_cfg["steps"]),
            "steps_planned": int(profile_cfg["steps"]),
            "steps_completed": int(profile_cfg["steps"]),
            "train_epochs": "not_applicable",
            "train_epochs_planned": "not_applicable",
            "train_epochs_completed": "not_applicable",
            "train_steps": int(profile_cfg["steps"]),
            "train_budget_policy": task_record["train_budget_policy"],
            "train_budget_unit": "steps",
            "train_budget_value": int(profile_cfg["steps"]),
            "completed_train_units": int(profile_cfg["steps"]),
            "train_examples_seen": int(profile_cfg["steps"]) * int(task_record["resolved_effective_batch_size"]),
            "train_tokens_seen": int(profile_cfg["steps"]) * int(task_record["resolved_effective_batch_size"]) * int(task_record["resolved_padded_sequence_length"]),
            "batch_size": task_record["resolved_batch_size"],
            "gradient_accumulation_steps": task_record["resolved_gradient_accumulation_steps"],
            "effective_batch_size": task_record["resolved_effective_batch_size"],
            "eval_batch_size": task_record["resolved_eval_batch_size"],
            "validation_eval_budget": int(profile_cfg["validation_examples"]),
            "test_eval_budget": int(profile_cfg["test_examples"]),
            "eval_batches": math.ceil(int(profile_cfg["test_examples"]) / int(task_record["resolved_eval_batch_size"])),
            "learning_rate": task_record["resolved_learning_rate"],
            "base_learning_rate": task_record["resolved_base_learning_rate"],
            "lr_scheduler": task_record["resolved_lr_scheduler"],
            "warmup_ratio": task_record["resolved_warmup_ratio"],
            "warmup_steps": max(1, int(round(int(profile_cfg["steps"]) * float(task_record["resolved_warmup_ratio"])))),
            "min_lr_ratio": task_record["resolved_min_lr_ratio"],
            "min_learning_rate": task_record["resolved_min_learning_rate"],
            "cosine_total_steps": int(profile_cfg["steps"]),
            "weight_decay": task_record["resolved_weight_decay"],
            "grad_clip_norm": task_record["resolved_grad_clip_norm"],
            "log_every": int(profile_cfg["log_every"]),
            "log_step_policy": task_record["resolved_log_step_policy"],
            "logging_reference_train_steps": int(profile_cfg["steps"]),
            "min_logged_train_step_count": int(profile_cfg["steps"]) if int(profile_cfg["steps"]) < 100 else math.ceil(int(profile_cfg["steps"]) * 0.01),
            "planned_logged_train_step_count": len({1, int(profile_cfg["steps"]), *range(int(profile_cfg["log_every"]), int(profile_cfg["steps"]) + 1, int(profile_cfg["log_every"]))}),
            "actual_logged_train_step_count": len(train_log_rows),
            "log_coverage_ratio": len(train_log_rows) / max(int(profile_cfg["steps"]), 1),
            "log_policy_satisfied": True,
            "eval_every": int(profile_cfg["eval_every"]),
            "checkpoint_every": 0,
            "checkpoint_policy": task_record["resolved_checkpoint_policy"],
            "checkpoint_manifest_path": str(run_dir / "checkpoint_manifest.json"),
            "attention_contract": ATTENTION_CONTRACT,
            "causal": False,
            "graph_directionality": GRAPH_DIRECTIONALITY,
            "attention_backend": backend,
            "graph_id": task_record["resolved_graph_id"],
            "graph_seed": task_record["resolved_graph_seed"],
            "graph_generation_algorithm": task_record["resolved_graph_generation_algorithm"],
            "canonical_graph_dir": str(Path(task_record["graph_artifacts"]["selected_graph_path"]).parent),
            "canonical_graph_artifact_path": task_record["graph_artifacts"]["selected_graph_path"],
            "canonical_graph_artifact_sha256": task_record["graph_artifacts"]["selected_graph_sha256"],
            "canonical_graph_seed": task_record["resolved_graph_seed"],
            "canonical_graph_generation_algorithm": task_record["resolved_graph_generation_algorithm"],
            "graph_generation_status": "ok",
            "graph_generation_attempts": 1,
            "graph_artifact_path": graph_paths["graph_artifact_path"],
            "graph_generation_path": graph_paths["graph_generation_path"],
            "graph_certificate_path": graph_paths["graph_certificate_path"],
            "selected_graph_sha256": graph_paths["graph_artifact_sha256"],
            "graph_artifact_sha256": graph_paths["graph_artifact_sha256"],
            "graph_artifact_sha256_matches_canonical": graph_paths["graph_artifact_sha256"] == task_record["graph_artifacts"]["selected_graph_sha256"],
            "graph_certificate_sha256": graph_paths["graph_certificate_sha256"],
            "graph_block_policy": task_record["graph_block_policy"],
            "graph_degree_or_budget_policy": task_record["graph_degree_or_budget_policy"],
            "graph_block_size": task_record["resolved_graph_block_size"],
            "graph_num_blocks_or_nodes": task_record["resolved_graph_num_blocks_or_nodes"],
            "graph_degree": task_record["resolved_graph_degree_or_budget"],
            "N_total": task_record["resolved_raw_sequence_length"],
            "B": task_record["resolved_graph_block_size"],
            "q": task_record["resolved_q_alias_if_applicable"],
            "d": task_record["resolved_graph_degree_or_budget"],
            "N_total_v07_alias": task_record["resolved_raw_sequence_length"],
            "B_v07_alias": task_record["resolved_graph_block_size"],
            "q_v07_alias": task_record["resolved_q_alias_if_applicable"],
            "d_v07_alias": task_record["resolved_graph_degree_or_budget"],
            "v07_alias_replacement_reason": "v08 probe uses task-specific raw sequence length under non_causal contract",
            "G_type": "permutation_regular",
            "H_type": "permutation_regular",
            "allow_multiedges": True,
            "multiplicity_mode": "unique_log_m",
            "lambda_G": cert.get("lambda_G", "not_applicable"),
            "mu_H": cert.get("mu_H", "not_applicable"),
            "rho_zigzag_bound": cert.get("rho_zigzag_bound", "not_applicable"),
            "rho_zigzag_certified": cert.get("rho_zigzag_certified", "not_applicable"),
            "rho_zigzag_exact": cert.get("rho_zigzag_exact", "not_applicable"),
            "rot_g_is_bijection": cert.get("rot_g_is_bijection", "not_applicable"),
            "P_G_row_stochastic_error": cert.get("P_G_row_stochastic_error", "not_applicable"),
            "P_G_col_stochastic_error": cert.get("P_G_col_stochastic_error", "not_applicable"),
            "P_H_row_stochastic_error": cert.get("P_H_row_stochastic_error", "not_applicable"),
            "P_H_col_stochastic_error": cert.get("P_H_col_stochastic_error", "not_applicable"),
            "graph_certified": cert.get("rho_zigzag_certified", False),
            "implementation_certified": bool(method == "zigzag_certified" and cert.get("rho_zigzag_certified", False)),
            "theory_aligned_method": bool(method == "zigzag_certified" and cert.get("rho_zigzag_certified", False)),
            "duplicate_rate": artifacts.metrics.get("duplicate_rate", "not_applicable"),
            "self_loop_rate": artifacts.metrics.get("self_loop_rate", "not_applicable"),
            "remote_local_overlap_mean": cert.get("remote_local_overlap_mean", "not_applicable"),
            "collision_count_mean": cert.get("collision_count_mean", "not_applicable"),
            "zigzag_actual_k_min_after_causal": "not_applicable",
            "zigzag_actual_k_mean_after_causal": "not_applicable",
            "zigzag_actual_k_max_after_causal": "not_applicable",
            "zigzag_attention_pair_count_after_causal": "not_applicable",
            "zigzag_actual_k_min_noncausal": zigzag_budget.get("actual_k_min_noncausal", "not_applicable"),
            "zigzag_actual_k_mean_noncausal": zigzag_budget.get("actual_k_mean_noncausal", "not_applicable"),
            "zigzag_actual_k_max_noncausal": zigzag_budget.get("actual_k_max_noncausal", "not_applicable"),
            "zigzag_attention_pair_count_noncausal": zigzag_budget.get("attention_pair_count_noncausal", "not_applicable"),
            "random_target_k_source": random_budget.get("random_target_k_source", "not_applicable"),
            "random_actual_k_min_after_causal": "not_applicable",
            "random_actual_k_mean_after_causal": "not_applicable",
            "random_actual_k_max_after_causal": "not_applicable",
            "random_attention_pair_count_after_causal": "not_applicable",
            "random_actual_k_min_noncausal": random_budget.get("actual_k_min_noncausal", "not_applicable"),
            "random_actual_k_mean_noncausal": random_budget.get("actual_k_mean_noncausal", "not_applicable"),
            "random_actual_k_max_noncausal": random_budget.get("actual_k_max_noncausal", "not_applicable"),
            "random_attention_pair_count_noncausal": random_budget.get("attention_pair_count_noncausal", "not_applicable"),
            "random_k_alignment_error_mean": random_budget.get("random_k_alignment_error_mean", "not_applicable"),
            "random_k_alignment_error_max": random_budget.get("random_k_alignment_error_max", "not_applicable"),
            "random_alignment_mode": random_budget.get("random_alignment_mode", "not_applicable"),
            "random_k_aligned_to_zigzag": random_budget.get("random_k_aligned_to_zigzag", "not_applicable"),
            "attention_pair_count_after_causal": "not_applicable",
            "attention_k_min": artifacts.metrics.get("effective_k_min", "not_applicable"),
            "attention_k_mean": artifacts.metrics.get("effective_k_mean", "not_applicable"),
            "attention_k_max": artifacts.metrics.get("effective_k_max", "not_applicable"),
            "attention_pair_count": artifacts.metrics.get("attention_pair_count", "not_applicable"),
            "attention_diagnostics_path": str(run_dir / "attention_diagnostics.json"),
            "zigzag_budget_path": str(run_dir / "zigzag_budget.json"),
            "random_budget_path": str(run_dir / "random_budget.json"),
            "train_loss_final": train_loss,
            "validation_loss_final": validation_result["loss"],
            "test_loss": test_result["loss"],
            "primary_metric_value": test_result["primary_metric_value"],
            "secondary_metrics_json": json_metric(test_result["secondary_metrics"]),
            "task_metrics_json": json_metric(test_result["task_metrics"]),
            "final_train_loss": train_loss,
            "test_perplexity": "not_applicable",
            "validation_perplexity_if_applicable": "not_applicable",
            "test_perplexity_if_applicable": "not_applicable",
            "train_tokens_per_sec": (int(profile_cfg["steps"]) * int(task_record["resolved_effective_batch_size"]) * int(task_record["resolved_padded_sequence_length"])) / elapsed,
            "validation_tokens_per_sec": validation_result["tokens_per_sec"],
            "test_tokens_per_sec": test_result["tokens_per_sec"],
            "train_examples_per_sec": (int(profile_cfg["steps"]) * int(task_record["resolved_effective_batch_size"])) / elapsed,
            "validation_examples_per_sec": validation_result["examples_per_sec"],
            "test_examples_per_sec": test_result["examples_per_sec"],
            "total_wall_time_sec": elapsed,
            "train_wall_time_sec": elapsed - validation_result["elapsed_sec"] - test_result["elapsed_sec"],
            "eval_wall_time_sec": validation_result["elapsed_sec"] + test_result["elapsed_sec"],
            "data_prep_wall_time_sec": "not_applicable",
            "seconds_since_prev_log_mean": sum(since_values) / max(len(since_values), 1),
            "seconds_since_prev_log_max": max(since_values) if since_values else 0.0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
            "oom_fallback_applied": False,
            "oom_fallback_reason": "not_applicable",
            "training_curves_path": str(run_dir / "training_curves.png"),
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "summary_path": str(run_dir / "summary.json"),
            "raw_config_snapshot_path": str(run_dir / "raw_config_snapshot.json"),
            "resolved_config_snapshot_path": str(run_dir / "resolved_config_snapshot.json"),
            "result_field_audit_path": str(run_dir / "result_field_audit.json"),
        }
    )
    task = task_record["task"]
    if task == "copy":
        row["copy_token_accuracy"] = test_result["secondary_metrics"].get("copy_token_accuracy", 0.0)
        row["copy_sequence_accuracy"] = test_result["secondary_metrics"].get("copy_sequence_accuracy", 0.0)
        row["copy_eos_accuracy"] = "not_applicable"
        row["copy_source_length"] = task_record["resolved_sequence_length_max"]
        row["eval_token_accuracy"] = row["copy_token_accuracy"]
        row["eval_sequence_accuracy"] = row["copy_sequence_accuracy"]
        row["eval_eos_accuracy"] = "not_applicable"
    elif task == "selective_copy":
        row["selective_copy_token_accuracy"] = test_result["secondary_metrics"].get("selective_copy_token_accuracy", 0.0)
        row["selective_copy_sequence_accuracy"] = test_result["secondary_metrics"].get("selective_copy_sequence_accuracy", 0.0)
        row["eval_token_accuracy"] = row["selective_copy_token_accuracy"]
        row["eval_sequence_accuracy"] = row["selective_copy_sequence_accuracy"]
        row["eval_eos_accuracy"] = "not_applicable"
    elif task in {"induction_associative_recall", "niah_kv_retrieval", "ruler"}:
        row["retrieval_exact_match"] = test_result["secondary_metrics"].get("retrieval_exact_match", test_result["secondary_metrics"].get("exact_match", 0.0))
        row["retrieval_token_accuracy"] = test_result["secondary_metrics"].get("retrieval_token_accuracy", test_result["secondary_metrics"].get("token_accuracy", 0.0))
        row["retrieval_answer_format"] = "position_value" if task == "induction_associative_recall" else "utf8_string"
        if task == "ruler":
            row["ruler_subtask"] = "all"
            row["ruler_subtask_exact_match"] = test_result["secondary_metrics"].get("ruler_subtask_exact_match", row["retrieval_exact_match"])
            row["ruler_subtask_token_accuracy"] = test_result["secondary_metrics"].get("ruler_subtask_token_accuracy", row["retrieval_token_accuracy"])
    elif task == "lra_listops":
        row["listops_accuracy"] = test_result["secondary_metrics"].get("listops_accuracy", test_result["secondary_metrics"].get("accuracy", 0.0))
        row["listops_macro_accuracy"] = test_result["secondary_metrics"].get("listops_macro_accuracy", test_result["secondary_metrics"].get("macro_accuracy", 0.0))
        row["listops_class_count"] = 10
    return row


def run_one(config: dict, manifest: dict, task: str, method: str, seed: int, device: torch.device) -> dict:
    records = task_records(manifest)
    task_record = records[task]
    profile = config["profile"]
    profile_cfg = run_profile(task_record, profile)
    phase = config["phase"]
    run_id = f"{task}_seed{seed}_{method}_{profile}"
    run_dir = Path(config["output_root"]) / task / method
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / "summary.json"
    if existing.exists():
        payload = read_json(existing)
        if payload.get("status") == "ok":
            return payload["result"]
    started = time.perf_counter()
    command = command_string()
    write_command(run_dir / "command.sh", command)
    write_json(run_dir / "raw_config_snapshot.json", config)
    resolved = copy.deepcopy(task_record)
    resolved.update({"method": method, "seed": seed, "phase": phase, "profile": profile})
    write_json(run_dir / "resolved_config_snapshot.json", resolved)
    write_json(run_dir / "config_snapshot.json", resolved)
    write_json(run_dir / "phase4_task_parameter_record.json", task_record)
    write_json(
        run_dir / "checkpoint_manifest.json",
        {
            "checkpoint_policy": task_record["resolved_checkpoint_policy"],
            "tensor_checkpoints_written": [],
            "external_checkpoint_path": "not_applicable",
            "retention_reason": "v08 first sweep records manifest only; no tensor checkpoint is written",
        },
    )
    graph_paths = copy_graph_artifacts(task_record, run_dir)
    split_stores = stores(task_record)
    encoder, model, artifacts, backend = build_model_and_artifacts(task_record, method, seed, device)
    args_for_zigzag = artifact_args(task_record, "zigzag_certified", seed)
    zigzag_backend = resolve_attention_backend(str(task_record["resolved_attention_backend"]), "zigzag_certified")
    zigzag_artifacts = make_attention_artifacts(
        "zigzag_certified",
        int(task_record["resolved_padded_sequence_length"]),
        args_for_zigzag,
        device,
        zigzag_backend,
    )
    zigzag_budget = budget_payload("zigzag_certified", zigzag_artifacts, task_record)
    random_budget = budget_payload(method, artifacts, task_record, zigzag_artifacts) if method == "random_regular" else budget_payload("random_regular", make_attention_artifacts("random_regular", int(task_record["resolved_padded_sequence_length"]), artifact_args(task_record, "random_regular", seed), device, resolve_attention_backend(str(task_record["resolved_attention_backend"]), "random_regular")), task_record, zigzag_artifacts)
    write_json(run_dir / "zigzag_budget.json", zigzag_budget)
    write_json(run_dir / "random_budget.json", random_budget)
    write_json(
        run_dir / "attention_diagnostics.json",
        {
            "method": method,
            "attention_contract": ATTENTION_CONTRACT,
            "causal": False,
            "graph_directionality": GRAPH_DIRECTIONALITY,
            "attention_backend": backend,
            "metrics": artifacts.metrics,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_record["resolved_base_learning_rate"]),
        weight_decay=float(task_record["resolved_weight_decay"]),
    )
    metrics_rows: list[dict] = []
    train_loss = math.nan
    prev_log = time.perf_counter()
    train_started = time.perf_counter()
    total_steps = int(profile_cfg["steps"])
    accum = int(task_record["resolved_gradient_accumulation_steps"])
    batch_size = int(task_record["resolved_batch_size"])
    for step in range(1, total_steps + 1):
        lr = schedule_lr(
            float(task_record["resolved_base_learning_rate"]),
            float(task_record["resolved_min_learning_rate"]),
            max(1, int(round(total_steps * float(task_record["resolved_warmup_ratio"])))),
            total_steps,
            step,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for accum_index in range(accum):
            raw_rows = split_stores["train"].sample(
                batch_size,
                seed,
                f"{task}:{method}:{profile}",
                (step - 1) * accum + accum_index,
                limit=int(profile_cfg["train_examples"]),
            )
            batch = make_probe_batch(raw_rows, task_record, encoder, device)
            loss, _batch_metrics, _per_sample = forward_loss_and_metrics(model, artifacts, batch, task_record)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step={step}")
            (loss / accum).backward()
            step_loss += float(loss.item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(task_record["resolved_grad_clip_norm"])))
        optimizer.step()
        train_loss = step_loss / accum
        should_log = step == 1 or step % int(profile_cfg["log_every"]) == 0 or step == total_steps
        if should_log:
            eval_result = evaluate(
                model,
                artifacts,
                encoder,
                task_record,
                split_stores["validation"],
                limit=int(profile_cfg.get("log_eval_examples", profile_cfg["validation_examples"])),
                batch_size=int(task_record["resolved_eval_batch_size"]),
                device=device,
            )
            now = time.perf_counter()
            metrics_rows.append(
                metric_row(
                    run_id,
                    task_record,
                    method,
                    seed,
                    step,
                    "train",
                    phase,
                    train_loss,
                    eval_result,
                    lr,
                    now - train_started,
                    now - prev_log,
                    grad_norm,
                )
            )
            prev_log = now
            print(json.dumps(metrics_rows[-1], ensure_ascii=False), flush=True)
    metrics_path = run_dir / "metrics.jsonl"
    write_jsonl(metrics_path, metrics_rows)
    (run_dir / f"{method}_metrics.jsonl").write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    validation_result = evaluate(
        model,
        artifacts,
        encoder,
        task_record,
        split_stores["validation"],
        limit=int(profile_cfg["validation_examples"]),
        batch_size=int(task_record["resolved_eval_batch_size"]),
        device=device,
    )
    test_result = evaluate(
        model,
        artifacts,
        encoder,
        task_record,
        split_stores["test"],
        limit=int(profile_cfg["test_examples"]),
        batch_size=int(task_record["resolved_eval_batch_size"]),
        device=device,
    )
    write_training_curves(metrics_rows, run_dir / "training_curves.png")
    row = build_result_row(
        task_record,
        profile_cfg,
        method,
        seed,
        phase,
        run_id,
        run_dir,
        command,
        backend,
        parameter_count(model),
        graph_paths,
        artifacts,
        zigzag_budget,
        random_budget,
        train_loss,
        validation_result,
        test_result,
        metrics_rows,
        started,
    )
    audit = audit_result_row(row, metrics_rows, task_record, {"task": task, "method": method, "attention_contract": ATTENTION_CONTRACT, "causal": False, "graph_directionality": GRAPH_DIRECTIONALITY})
    row["log_policy_satisfied"] = audit["log_policy_satisfied"]
    write_json(run_dir / "result_field_audit.json", audit)
    write_csv(run_dir / "results.csv", [row], FIELD_CONTRACT)
    write_jsonl(run_dir / "results.jsonl", [row])
    result = {
        "status": "ok",
        "run_id": run_id,
        "task": task,
        "method": method,
        "profile": profile,
        "result": row,
        "validation": validation_result,
        "test": test_result,
        "total_wall_time_sec": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", result)
    return row


def aggregate(config: dict) -> dict:
    root = Path(config["output_root"])
    rows = []
    audits = []
    for task in config["tasks"]:
        for method in config["methods"]:
            result_path = root / task / method / "results.jsonl"
            audit_path = root / task / method / "result_field_audit.json"
            if result_path.exists():
                rows.extend(json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip())
            if audit_path.exists():
                audits.append(read_json(audit_path))
    if rows:
        write_csv(root / "results_all.csv", rows, FIELD_CONTRACT)
        write_jsonl(root / "results_all.jsonl", rows)
        write_csv(root / "results.csv", rows, FIELD_CONTRACT)
        write_jsonl(root / "results.jsonl", rows)
    summary = {
        "version": EXPERIMENT_VERSION,
        "phase": config["phase"],
        "profile": config["profile"],
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "partial_or_failed",
        "expected_runs": len(config["tasks"]) * len(config["methods"]) * len(config.get("seeds", [0])),
        "completed_runs": len(rows),
        "result_field_audits_passed": sum(1 for audit in audits if audit.get("status") == "passed"),
        "result_field_audits_total": len(audits),
        "timestamp_utc": utc_now(),
    }
    write_json(root / "summary.json", summary)
    write_json(
        root / "result_field_audit.json",
        {
            "status": "passed" if audits and all(audit.get("status") == "passed" for audit in audits) else "failed",
            "run_audits": audits,
        },
    )
    return summary


def task_records(manifest: dict) -> dict[str, dict]:
    return {row["task"]: row for row in manifest["tasks"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(config)
    device = select_device(args.device)
    if args.aggregate_only:
        aggregate(config)
        return
    tasks = [args.task] if args.task else list(config["tasks"])
    methods = [args.method] if args.method else list(config["methods"])
    seeds = [args.seed] if args.seed is not None else list(config.get("seeds", [0]))
    for task in tasks:
        for method in methods:
            for seed in seeds:
                run_one(config, manifest, task, method, int(seed), device)
    aggregate(config)


if __name__ == "__main__":
    main()
