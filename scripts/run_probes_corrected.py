from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
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
from probe_metrics import aggregate_metric_rows, json_metric, write_training_curves
from probe_tasks import JsonlStore, ProbeTransformer, load_encoder, make_probe_batch, parameter_count
from run_probe_experiment import forward_loss_and_metrics, schedule_lr
from synthetic_mvp_core.artifacts import (
    build_random_remote_rows_aligned_to_zigzag_noncausal,
    make_attention_artifacts,
    resolve_attention_backend,
)


VERSION = "probes_corrected_valid_as_test_l8_log5"
BRANCH = "codex/probes-corrected-valid-as-test-l8-log5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_all_seeds(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "python_random_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "torch_cuda_manual_seed_all": bool(torch.cuda.is_available()),
        "torch_cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else False,
        "torch_cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch.backends, "cudnn") else False,
        "torch_deterministic_algorithms": False,
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


def task_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task"]): row for row in manifest["tasks"]}


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
    rng = random.Random(f"{VERSION}|data|{int(data_seed)}|epoch|{int(epoch)}")
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


def run_dir_for(config: dict[str, Any], task: str, method: str, seed: int, mode: str = "train") -> Path:
    trial = str(config.get("trial_id", "main_l8_log5"))
    if mode == "smoke":
        return Path(config["output_root"]) / trial / "smoke" / task / method / f"seed{int(seed)}"
    return Path(config["output_root"]) / trial / task / method / f"seed{int(seed)}"


def run_identity(config_path: Path, config: dict[str, Any], manifest_path: Path, record: dict[str, Any], task: str, method: str, seed: int) -> dict[str, Any]:
    graph = record["graph_artifacts"]
    return {
        "version": VERSION,
        "trial_id": config.get("trial_id", "main_l8_log5"),
        "task": task,
        "method": method,
        "seed": int(seed),
        "branch_name": BRANCH,
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
        "discarded_old_test_sha256": record["discarded_old_test_sha256"],
        "graph_sha256": graph["selected_graph_sha256"],
        "position_encoding": record["position_encoding"],
        "vocab_size": record["resolved_vocab_or_value_space_size"],
        "token_output_size": record["resolved_token_output_size"],
        "target_position_policy": record["input_contract"]["target_position_policy"],
        "T": record["resolved_padded_sequence_length"],
    }


def identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def artifact_args(record: dict[str, Any], method: str, seed: int) -> SimpleNamespace:
    graph = record["graph_artifacts"]["artifact"]
    return SimpleNamespace(
        block_size=int(record["resolved_graph_block_size"]),
        degree=int(record["resolved_graph_degree_or_budget"]),
        causal=False,
        graph_config=graph,
        graph_artifact=graph,
        graph_certificate=record["graph_artifacts"]["certificate"],
        graph_artifact_path=record["graph_artifacts"]["selected_graph_path"],
        seed=int(seed),
        random_alignment_mode="per_query_noncausal_unique_k",
        random_target_k_source="zigzag_actual_noncausal_per_query_unique_k",
        multiplicity_mode="unique_log_m",
    )


def method_artifact_args(record: dict[str, Any], method: str, seed: int) -> SimpleNamespace:
    args = artifact_args(record, method, seed)
    if method == "random_regular":
        args.random_aligned_rows = build_random_remote_rows_aligned_to_zigzag_noncausal(
            int(record["resolved_padded_sequence_length"]),
            args,
        )
    return args


def build_model(record: dict[str, Any], method: str, seed: int, device: torch.device):
    backend = resolve_attention_backend(str(record["resolved_attention_backend"]), method)
    args = method_artifact_args(record, method, seed)
    artifacts = make_attention_artifacts(method, int(record["resolved_padded_sequence_length"]), args, device, backend)
    seed_policy = set_all_seeds(int(record.get("model_seed", seed)))
    use_class_head = str(record["resolved_loss_type"]) == "classification_cross_entropy"
    model = ProbeTransformer(
        vocab_size=int(record["resolved_vocab_or_value_space_size"]),
        token_output_size=int(record["resolved_token_output_size"]),
        class_count=10,
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
        use_class_head=use_class_head,
    ).to(device)
    return model, artifacts, backend, seed_policy


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


def per_sample_loss_sum(per_sample: list[dict[str, Any]], batch_loss: torch.Tensor) -> float:
    if per_sample and all("loss_sum" in item for item in per_sample):
        return sum(float(item["loss_sum"]) for item in per_sample)
    return float(batch_loss.item()) * max(len(per_sample), 1)


def per_sample_token_count(per_sample: list[dict[str, Any]]) -> int:
    return sum(int(item.get("tokens", item.get("examples", 1))) for item in per_sample)


def make_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    record: dict[str, Any],
    identity: dict[str, Any],
    epoch: int,
    optimizer_step: int,
    permutation_position: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {"type": record["resolved_lr_scheduler"]},
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
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


def profile_record(record: dict[str, Any], mode: str) -> dict[str, Any]:
    return dict(record["smoke" if mode == "smoke" else "main"])


def train_loop(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    task: str,
    method: str,
    seed: int,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, task, method, seed, mode=mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config_path, config, manifest_path, record, task, method, seed)
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

    data_dir = Path(record["version_path"])
    if (data_dir / "validation.jsonl").exists():
        raise RuntimeError(f"{task}: validation.jsonl is forbidden in corrected valid-as-test data")
    train_store = JsonlStore(data_dir / "train.jsonl")
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    profile = profile_record(record, mode)
    model, artifacts, backend, seed_policy = build_model(record, method, seed, device)
    initial_hash = state_dict_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(record["resolved_base_learning_rate"]),
        weight_decay=float(record["resolved_weight_decay"]),
    )

    max_steps = int(profile["steps"])
    batch_size = int(record["resolved_batch_size"])
    accum = int(record["resolved_gradient_accumulation_steps"])
    base_lr = float(record["resolved_base_learning_rate"])
    checkpoint_every = int(profile.get("checkpoint_every", 0))
    epochs = int(profile.get("epochs", 1))
    log_every = int(profile.get("log_every", 5))
    diagnostic_rows = [train_store.row(i) for i in range(min(int(profile.get("train_diagnostic_examples", 16)), len(train_store)))]

    metrics_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    train_loss_last = math.nan
    train_loss_sum = 0.0
    train_token_count = 0
    global_step = 0
    started = time.perf_counter()
    prev_log = started
    model.train()

    for epoch in range(epochs):
        permutation = deterministic_permutation(len(train_store), int(record.get("data_seed", seed)), epoch)
        coverage = epoch_coverage(permutation, len(train_store))
        coverage["epoch"] = epoch
        coverage_rows.append(coverage)
        position = 0
        while position < len(permutation) and global_step < max_steps:
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
            micro_batches = 0
            for _micro in range(accum):
                batch_indices = permutation[position : position + batch_size]
                if not batch_indices:
                    break
                position += len(batch_indices)
                raw_rows = [train_store.row(index) for index in batch_indices]
                batch = make_probe_batch(raw_rows, record, encoder, device)
                loss, _metrics, per_sample = forward_loss_and_metrics(model, artifacts, batch, record)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step={global_step}")
                (loss / accum).backward()
                step_loss_sum += per_sample_loss_sum(per_sample, loss)
                step_tokens += per_sample_token_count(per_sample)
                micro_batches += 1
            if micro_batches == 0:
                break
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(record["resolved_grad_clip_norm"])))
            optimizer.step()
            train_loss_last = step_loss_sum / max(step_tokens, 1)
            train_loss_sum += step_loss_sum
            train_token_count += step_tokens
            if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
                diag = evaluate_rows(model, artifacts, encoder, record, diagnostic_rows, int(record["resolved_eval_batch_size"]), device)
                now = time.perf_counter()
                row = {
                    "run_id": f"{VERSION}_{config.get('trial_id', 'main_l8_log5')}_{task}_{method}_seed{seed}",
                    "mode": mode,
                    "task": task,
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
                    "primary_metric_name": record["primary_metric"],
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
                        record,
                        identity,
                        epoch,
                        global_step,
                        position,
                    )
                )

    final_checkpoint = make_checkpoint(
        run_dir / "checkpoints" / f"{mode}_final_step{global_step}.pt",
        model,
        optimizer,
        record,
        identity,
        epochs - 1 if epochs else 0,
        global_step,
        0,
    )
    checkpoints.append(final_checkpoint)
    write_checkpoint_manifest(run_dir, checkpoints, "tensor_checkpoint_with_optimizer_sampler_rng_state")
    write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    write_training_curves(metrics_rows, run_dir / "training_curves.png")
    if coverage_rows:
        write_json(run_dir / "sampler_coverage.json", {"epochs": coverage_rows})
    summary = {
        "status": "ok",
        "mode": mode,
        "version": VERSION,
        "run_id": f"{VERSION}_{config.get('trial_id', 'main_l8_log5')}_{task}_{method}_seed{seed}",
        "task": task,
        "method": method,
        "seed": int(seed),
        "identity": identity,
        "identity_sha256": identity_hash,
        "command": command,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "backend": backend,
        "seed_policy": seed_policy,
        "model": {
            "layers": record["resolved_layers"],
            "d_model": record["resolved_d_model"],
            "heads": record["resolved_heads"],
            "ffn_dim": record["resolved_ffn_dim"],
            "dropout": record["resolved_dropout"],
        },
        "initial_model_state_sha256": initial_hash,
        "parameter_count": parameter_count(model),
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "train_loss_last_step": train_loss_last,
        "train_loss_mean": train_loss_sum / max(train_token_count, 1),
        "steps_completed": global_step,
        "checkpoint": final_checkpoint,
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "training_curves_path": str(run_dir / "training_curves.png"),
        "checkpoint_manifest_path": str(run_dir / "checkpoint_manifest.json"),
        "test_read_during_training": False,
        "validation_read_during_training": False,
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


def final_eval(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    record: dict[str, Any],
    task: str,
    method: str,
    seed: int,
    device: torch.device,
    checkpoint: Path | None,
    mode: str,
) -> dict[str, Any]:
    run_dir = run_dir_for(config, task, method, seed, mode=mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        ckpt_manifest = read_json(run_dir / "checkpoint_manifest.json")
        checkpoint = Path(ckpt_manifest["latest_checkpoint"]["path"])
    identity = run_identity(config_path, config, manifest_path, record, task, method, seed)
    model, artifacts, backend, seed_policy = build_model(record, method, seed, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("identity_sha256") != identity_sha256(identity):
        raise RuntimeError("checkpoint identity does not match current config/data/graph/code")
    model.load_state_dict(payload["model_state"])
    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    first_test_read_at = utc_now()
    test_store = JsonlStore(Path(record["version_path"]) / "test.jsonl")
    profile = profile_record(record, mode)
    limit = min(len(test_store), int(profile.get("test_examples", len(test_store))))
    test_rows = [test_store.row(i) for i in range(limit)]
    result = evaluate_rows(model, artifacts, encoder, record, test_rows, int(record["resolved_eval_batch_size"]), device)
    out = {
        "status": "ok",
        "mode": "final-eval",
        "version": VERSION,
        "run_id": f"{VERSION}_{config.get('trial_id', 'main_l8_log5')}_{task}_{method}_seed{seed}_final_eval",
        "task": task,
        "method": method,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": tensor_checkpoint_sha256(checkpoint),
        "checkpoint_identity_sha256": payload.get("identity_sha256"),
        "identity": identity,
        "identity_sha256": identity_sha256(identity),
        "backend": backend,
        "seed_policy": seed_policy,
        "first_test_read_at": first_test_read_at,
        "test_source": "source_validation_jsonl",
        "source_test_discarded_sha256": record["discarded_old_test_sha256"],
        "test_examples": result["examples"],
        "test_target_tokens": result["tokens"],
        "test_loss": result["loss"],
        "primary_metric_name": record["primary_metric"],
        "primary_metric_value": result["primary_metric_value"],
        "secondary_metrics": result["secondary_metrics"],
        "task_metrics": result.get("task_metrics", {}),
        "position_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "pos" in name.lower()),
        "parameter_count": parameter_count(model),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "train_sha256": record["resolved_train_split_sha256"],
        "test_sha256": record["resolved_test_split_sha256"],
        "graph_sha256": record["graph_artifacts"]["selected_graph_sha256"],
        "target_position_policy": record["input_contract"]["target_position_policy"],
        "input_length": record["resolved_runtime_input_length"],
        "T": record["resolved_padded_sequence_length"],
        "padding_positions": record.get("runtime_padding_positions", 0),
    }
    if task == "selective_copy":
        out["selective_copy_token_accuracy"] = result["secondary_metrics"].get("selective_copy_token_accuracy", 0.0)
        out["selective_copy_sequence_accuracy"] = result["secondary_metrics"].get("selective_copy_sequence_accuracy", 0.0)
    elif task == "induction_associative_recall":
        out["retrieval_exact_match"] = result["secondary_metrics"].get("retrieval_exact_match", result["secondary_metrics"].get("exact_match", 0.0))
        out["retrieval_token_accuracy"] = result["secondary_metrics"].get("retrieval_token_accuracy", result["secondary_metrics"].get("token_accuracy", 0.0))
    elif task == "lra_listops":
        out["listops_accuracy"] = result["secondary_metrics"].get("listops_accuracy", result["secondary_metrics"].get("accuracy", 0.0))
        out["listops_macro_accuracy"] = result["secondary_metrics"].get("listops_macro_accuracy", result["secondary_metrics"].get("macro_accuracy", 0.0))
    write_json(run_dir / "final_eval.json", out)
    fieldnames = sorted(k for k, value in out.items() if not isinstance(value, (dict, list)))
    with (run_dir / "final_eval.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: out.get(key) for key in fieldnames})
    return out


def aggregate(config: dict[str, Any], manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    records = task_records(manifest)
    rows = []
    for task in config["tasks"]:
        for method in config["methods"]:
            for seed in config.get("seeds", [0]):
                run_dir = run_dir_for(config, task, method, int(seed), mode=mode)
                final_path = run_dir / "final_eval.json"
                summary_path = run_dir / "summary.json"
                if mode == "smoke":
                    final = {}
                    summary = read_json(summary_path) if summary_path.exists() else {}
                elif final_path.exists():
                    final = read_json(final_path)
                    summary = read_json(summary_path) if summary_path.exists() else {}
                else:
                    final = {}
                    summary = {}
                if summary.get("status") == "ok":
                    metrics_path = Path(str(summary.get("metrics_path", "")))
                    last_primary = summary.get("last_primary_metric_value")
                    row = {
                        "version": VERSION,
                        "mode": mode,
                        "task": task,
                        "method": method,
                        "seed": int(seed),
                        "status": final.get("status", summary.get("status")),
                        "primary_metric_name": final.get("primary_metric_name", records[task]["primary_metric"]),
                        "primary_metric_value": final.get("primary_metric_value", last_primary),
                        "test_loss": final.get("test_loss"),
                        "test_examples": final.get("test_examples"),
                        "test_target_tokens": final.get("test_target_tokens"),
                        "steps_completed": summary.get("steps_completed"),
                        "train_loss_mean": summary.get("train_loss_mean"),
                        "train_loss_last_step": summary.get("train_loss_last_step"),
                        "parameter_count": final.get("parameter_count", summary.get("parameter_count")),
                        "position_parameter_count": final.get("position_parameter_count", summary.get("position_parameter_count")),
                        "input_length": records[task]["resolved_runtime_input_length"],
                        "T": records[task]["resolved_padded_sequence_length"],
                        "layers": records[task]["resolved_layers"],
                        "d_model": records[task]["resolved_d_model"],
                        "heads": records[task]["resolved_heads"],
                        "ffn_dim": records[task]["resolved_ffn_dim"],
                        "batch_size": records[task]["resolved_batch_size"],
                        "gradient_accumulation_steps": records[task]["resolved_gradient_accumulation_steps"],
                        "effective_batch_size": records[task]["resolved_effective_batch_size"],
                        "log_every": records[task][mode if mode == "smoke" else "main"]["log_every"],
                        "train_sha256": final.get("train_sha256", records[task]["resolved_train_split_sha256"]),
                        "test_sha256": final.get("test_sha256", records[task]["resolved_test_split_sha256"]),
                        "source_test_discarded_sha256": final.get("source_test_discarded_sha256", records[task]["discarded_old_test_sha256"]),
                        "graph_sha256": final.get("graph_sha256", records[task]["graph_artifacts"]["selected_graph_sha256"]),
                        "metrics_path": summary.get("metrics_path"),
                        "training_curves_path": summary.get("training_curves_path"),
                        "final_eval_path": str(final_path) if final_path.exists() else "not_applicable",
                        "test_read_during_training": summary.get("test_read_during_training"),
                        "validation_read_during_training": summary.get("validation_read_during_training"),
                    }
                    row.update({k: v for k, v in final.items() if k.endswith("_accuracy") or k.endswith("_exact_match")})
                    rows.append(row)
    root = Path(config["output_root"]) / str(config.get("trial_id", "main_l8_log5"))
    root.mkdir(parents=True, exist_ok=True)
    if rows:
        write_jsonl(root / f"results_{mode}.jsonl", rows)
        fieldnames = sorted({key for row in rows for key in row})
        with (root / f"results_{mode}.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "version": VERSION,
        "mode": mode,
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "partial_or_failed",
        "expected_runs": len(config["tasks"]) * len(config["methods"]) * len(config.get("seeds", [0])),
        "completed_runs": len(rows),
        "blocked_tasks": manifest.get("blocked_tasks", []),
        "timestamp_utc": utc_now(),
        "results_csv": str(root / f"results_{mode}.csv"),
        "results_jsonl": str(root / f"results_{mode}.jsonl"),
    }
    write_json(root / f"summary_{mode}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/probes_corrected_valid_as_test_l8_log5.json"))
    parser.add_argument("--mode", choices=["smoke", "train", "final-eval", "aggregate"], required=True)
    parser.add_argument("--task")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--aggregate-mode", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["task_parameter_manifest"])
    manifest = load_manifest(config)
    records = task_records(manifest)
    device = select_device(args.device)
    if args.mode == "aggregate":
        print(json.dumps(aggregate(config, manifest, args.aggregate_mode), ensure_ascii=False, sort_keys=True))
        return
    tasks = [args.task] if args.task else list(config.get("tasks", records.keys()))
    methods = [args.method] if args.method else list(config.get("methods", ["local"]))
    seeds = [args.seed] if args.seed is not None else list(config.get("seeds", [0]))
    outputs = []
    for task in tasks:
        if task not in records:
            raise ValueError(f"task {task!r} is not trainable in corrected manifest")
        for method in methods:
            for seed in seeds:
                if args.mode in {"smoke", "train"}:
                    outputs.append(
                        train_loop(
                            config_path=args.config,
                            config=config,
                            manifest_path=manifest_path,
                            record=records[task],
                            task=task,
                            method=method,
                            seed=int(seed),
                            device=device,
                            mode=args.mode,
                        )
                    )
                else:
                    train_mode = "train"
                    outputs.append(
                        final_eval(
                            config_path=args.config,
                            config=config,
                            manifest_path=manifest_path,
                            record=records[task],
                            task=task,
                            method=method,
                            seed=int(seed),
                            device=device,
                            checkpoint=args.checkpoint,
                            mode=train_mode,
                        )
                    )
    print(json.dumps({"status": "ok", "mode": args.mode, "runs": outputs}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
