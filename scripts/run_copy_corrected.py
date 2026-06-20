from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

from probe_common import command_string, file_sha256, git_commit, git_dirty, read_json, write_command, write_json, write_jsonl
from probe_metrics import aggregate_metric_rows, json_metric
from probe_tasks import JsonlStore, ProbeTransformer, load_encoder, make_probe_batch, parameter_count
from run_probe_experiment import forward_loss_and_metrics, schedule_lr
from synthetic_mvp_core.artifacts import (
    build_random_remote_rows_aligned_to_zigzag_noncausal,
    build_random_remote_rows_for_actual_mask_density,
    make_attention_artifacts,
    resolve_attention_backend,
)


VERSION = "copy_corrected_v01_l8_log5"
BRANCH = "codex/copy-corrected-v01-l8-log5"


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


def state_dict_sha256(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def tensor_checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path)


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


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deterministic_permutation(n: int, data_seed: int, epoch: int) -> list[int]:
    indices = list(range(int(n)))
    rng = random.Random(f"copy_corrected_v01|data|{int(data_seed)}|epoch|{int(epoch)}")
    rng.shuffle(indices)
    return indices


def epoch_coverage(indices: list[int], n: int) -> dict[str, int]:
    counts = Counter(indices)
    return {
        "draw_count": len(indices),
        "unique_count": len(counts),
        "never_seen": int(n) - len(counts),
        "max_repeat_count": max(counts.values()) if counts else 0,
    }


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


def run_identity(config_path: Path, config: dict[str, Any], manifest_path: Path, record: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    graph = record["graph_artifacts"]
    random_actual_mask_density = config_random_density(config)
    random_layerwise_independent = config_random_layerwise_independent(config)
    return {
        "version": experiment_version(config, record),
        "trial_id": config.get("trial_id", "gate"),
        "method": method,
        "seed": int(seed),
        "random_actual_mask_density": random_actual_mask_density if method == "random_regular" else None,
        "random_layerwise_independent_masks": random_layerwise_independent if method == "random_regular" else None,
        "random_layerwise_mask_count": int(record["resolved_layers"]) if method == "random_regular" and random_layerwise_independent else None,
        "branch_name": experiment_branch(record),
        "branch_head_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
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


def identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    if method == "random_regular":
        config = config or {}
        density = config_random_density(config)
        if density is None:
            args.random_aligned_rows = build_random_remote_rows_aligned_to_zigzag_noncausal(
                int(record["resolved_padded_sequence_length"]),
                args,
            )
        else:
            args.random_actual_mask_density = float(density)
            args.random_alignment_mode = "actual_mask_density"
            args.random_target_k_source = "configured_actual_mask_density"
            args.random_aligned_rows = build_random_remote_rows_for_actual_mask_density(
                int(record["resolved_padded_sequence_length"]),
                args,
                float(density),
            )
    return args


def aggregate_layerwise_artifacts(layer_artifacts: list[Any]) -> SimpleNamespace:
    first = layer_artifacts[0]
    metrics = dict(first.metrics)
    per_layer_metrics = [dict(item.metrics) for item in layer_artifacts]
    pair_counts = [int(item.metrics.get("attention_pair_count", 0)) for item in layer_artifacts]
    unique_k_means = [float(item.metrics.get("unique_k_mean", 0.0)) for item in layer_artifacts]
    metrics.update(
        {
            "layerwise_independent_masks": True,
            "layerwise_mask_count": len(layer_artifacts),
            "per_layer_attention_pair_count": pair_counts,
            "per_layer_unique_k_mean": unique_k_means,
            "attention_pair_count_min_across_layers": min(pair_counts) if pair_counts else 0,
            "attention_pair_count_max_across_layers": max(pair_counts) if pair_counts else 0,
            "unique_k_mean_min_across_layers": min(unique_k_means) if unique_k_means else 0.0,
            "unique_k_mean_max_across_layers": max(unique_k_means) if unique_k_means else 0.0,
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
        metrics=metrics,
    )


def build_model(
    record: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
    config: dict[str, Any] | None = None,
):
    backend = resolve_attention_backend(str(record["resolved_attention_backend"]), method)
    if method == "random_regular" and config_random_layerwise_independent(config or {}):
        layer_artifacts = []
        for layer_index in range(int(record["resolved_layers"])):
            layer_args = method_artifact_args(record, method, seed, config, layer_index=layer_index)
            layer_artifacts.append(
                make_attention_artifacts(
                    method,
                    int(record["resolved_padded_sequence_length"]),
                    layer_args,
                    device,
                    backend,
                )
            )
        artifacts = aggregate_layerwise_artifacts(layer_artifacts)
    else:
        args = method_artifact_args(record, method, seed, config)
        artifacts = make_attention_artifacts(method, int(record["resolved_padded_sequence_length"]), args, device, backend)
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


def evaluate_rows(model, artifacts, encoder, record: dict[str, Any], raw_rows: list[dict[str, Any]], batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(raw_rows), batch_size):
            batch_rows = raw_rows[start_idx : start_idx + batch_size]
            batch = make_probe_batch(batch_rows, record, encoder, device)
            _loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record)
            rows.extend(per_sample)
    elapsed = max(time.perf_counter() - start, 1e-9)
    agg = aggregate_metric_rows(rows, record["primary_metric"])
    agg["elapsed_sec"] = elapsed
    agg["examples_per_sec"] = agg["examples"] / elapsed
    agg["tokens_per_sec"] = agg["tokens"] / elapsed
    model.train()
    return agg


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


def write_checkpoint_manifest(run_dir: Path, checkpoints: list[dict[str, Any]], policy: str) -> None:
    write_json(
        run_dir / "checkpoint_manifest.json",
        {
            "checkpoint_policy": policy,
            "tensor_checkpoints_written": checkpoints,
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
            "checkpoint_files_git_ignored": True,
        },
    )


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

    command = command_string()
    write_command(run_dir / "command.sh", command)
    write_json(run_dir / "raw_config_snapshot.json", config)
    write_json(run_dir / "resolved_config_snapshot.json", record)
    write_json(run_dir / "run_identity.json", identity)

    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_cfg = dict(config.get("train", {}))
    overfit = dict(config.get("gate_overfit", {}))
    active_record = gate_model_record(record, overfit) if mode == "gate-overfit" else record
    model, artifacts, backend, seed_policy = build_model(active_record, method, seed, device, config)
    initial_hash = state_dict_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(record["resolved_base_learning_rate"]),
        weight_decay=float(record["resolved_weight_decay"]),
    )
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
        max_steps = int(math.ceil(len(train_store) / int(record["resolved_effective_batch_size"]))) * int(train_cfg.get("epochs", 1))
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
            loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at gate-overfit step {step}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(record["resolved_grad_clip_norm"])))
            optimizer.step()
            train_loss_last = float(loss.item())
            train_loss_sum += sum(float(item["loss_sum"]) for item in per_sample)
            train_token_count += sum(int(item["tokens"]) for item in per_sample)
            diag = evaluate_rows(model, artifacts, encoder, record, fixed_rows, batch_size, device)
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
                lr = schedule_lr(
                    str(record["resolved_lr_scheduler"]),
                    base_lr,
                    float(record["resolved_min_learning_rate"]),
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
                    loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at step={global_step}")
                    (loss / accum).backward()
                    step_loss_sum += sum(float(item["loss_sum"]) for item in per_sample)
                    step_tokens += sum(int(item["tokens"]) for item in per_sample)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(record["resolved_grad_clip_norm"])))
                optimizer.step()
                train_loss_last = step_loss_sum / max(step_tokens, 1)
                train_loss_sum += step_loss_sum
                train_token_count += step_tokens
                if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
                    diag = evaluate_rows(model, artifacts, encoder, record, diagnostic_rows, int(record["resolved_eval_batch_size"]), device)
                    now = time.perf_counter()
                    row = {
                        "run_id": run_id_for(config, record, method, seed),
                        "mode": mode,
                        "task": "copy",
                        "method": method,
                        "seed": int(seed),
                        "step": global_step,
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
    write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
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
        "parameter_count": parameter_count(model),
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "train_loss_last_step": train_loss_last,
        "train_loss_mean": train_loss_sum / max(train_token_count, 1),
        "steps_completed": global_step,
        "checkpoint": final_checkpoint,
        "metrics_path": str(run_dir / "metrics.jsonl"),
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
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("identity_sha256") != identity_sha256(identity):
        raise RuntimeError("checkpoint identity does not match current config/data/graph/code")
    model.load_state_dict(payload["model_state"])
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    train_store = JsonlStore(Path(record["version_path"]) / "train.jsonl")
    baselines = train_baselines(train_store)
    first_test_read_at = utc_now()
    test_store = JsonlStore(Path(record["version_path"]) / "test.jsonl")
    test_rows = [test_store.row(i) for i in range(len(test_store))]
    result = evaluate_rows(model, artifacts, encoder, record, test_rows, int(record["resolved_eval_batch_size"]), device)
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
        "identity_sha256": identity_sha256(identity),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/copy_corrected_v01.json"))
    parser.add_argument("--mode", choices=["gate-overfit", "train", "final-eval"], required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
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
            else:
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
    print(json.dumps({"status": "ok", "mode": args.mode, "runs": outputs}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
