from __future__ import annotations

import copy
import gc
import json
import os
import random
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graph_diagnostics import compute_shortcut_stats
from graph_structures import DEFAULT_GRAPH_CONFIG, canonical_method
from v07_artifacts import materialize_graph_artifact

from .artifacts import (
    RESULT_FIELDS,
    AttentionArtifacts,
    budget_diagnostics,
    make_attention_artifacts,
    method_certification_fields,
    resolve_attention_backend,
)
from .common import set_seed
from .data import CopyBatch, TaskSpec, make_batch, padded_copy_lengths
from .io_utils import plot_training_curves, write_csv, write_json, write_jsonl
from .model import Transformer


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

def checkpoint_rng_state(device: torch.device) -> dict:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state

def copy_loss_and_metrics(logits: torch.Tensor, batch: CopyBatch) -> tuple[torch.Tensor, dict]:
    selected_logits = logits[:, batch.loss_positions, :]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        batch.targets.reshape(-1),
    )
    pred = selected_logits.argmax(dim=-1)
    correct = pred == batch.targets
    token_accuracy = float(correct.float().mean().item())
    sequence_accuracy = float(correct.all(dim=1).float().mean().item())
    eos_accuracy = float((pred[:, -1] == batch.targets[:, -1]).float().mean().item())
    return loss, {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "eos_accuracy": eos_accuracy,
    }

def evaluate(
    model,
    artifacts: AttentionArtifacts,
    args,
    spec,
    device,
    N: int,
    stream: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_token_correct = 0
    total_sequences = 0
    total_sequence_correct = 0
    total_eos_correct = 0
    with torch.no_grad():
        eval_batches = int(args.eval_batches)
        for batch_idx in range(eval_batches):
            batch = make_batch(
                args,
                spec,
                device,
                seq_len=N,
                batch_size=getattr(args, "eval_batch_size", args.batch_size),
                batch_index=batch_idx,
                stream=stream,
            )
            logits = model(
                batch.tokens,
                artifacts.mask,
                artifacts.local_valid,
                artifacts.neighbors,
                artifacts.valid_neighbors,
                artifacts.block_pair_index,
                artifacts.local_log_m,
                artifacts.neighbor_log_m,
            )
            loss, batch_metrics = copy_loss_and_metrics(logits, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite eval loss for {stream}")
            total_loss += float(loss.item())
            total_tokens += int(batch.targets.numel())
            total_token_correct += int(round(batch_metrics["token_accuracy"] * batch.targets.numel()))
            total_sequences += int(batch.targets.shape[0])
            total_sequence_correct += int(
                round(batch_metrics["sequence_accuracy"] * batch.targets.shape[0])
            )
            total_eos_correct += int(round(batch_metrics["eos_accuracy"] * batch.targets.shape[0]))
    model.train()
    return {
        "loss": total_loss / eval_batches,
        "token_accuracy": total_token_correct / total_tokens,
        "sequence_accuracy": total_sequence_correct / total_sequences,
        "eos_accuracy": total_eos_correct / total_sequences,
    }

def method_schedule_config(args, method: str) -> dict:
    overrides = dict(getattr(args, "method_overrides", {}).get(method, {}))
    scheduler = str(overrides.get("lr_scheduler", getattr(args, "lr_scheduler", "constant")))
    total_steps = overrides.get("cosine_total_steps", args.steps)
    if total_steps == "train_total_steps":
        total_steps = args.steps
    warmup_ratio = float(overrides.get("warmup_ratio", getattr(args, "warmup_ratio", 0.0)))
    min_lr_ratio = float(overrides.get("min_lr_ratio", getattr(args, "min_lr_ratio", 0.0)))
    base_lr = float(overrides.get("base_learning_rate", getattr(args, "base_learning_rate", args.learning_rate)))
    if "learning_rate" in overrides:
        base_lr = float(overrides["learning_rate"])
    warmup_steps = int(round(warmup_ratio * int(total_steps))) if scheduler == "cosine" else 0
    min_learning_rate = base_lr * min_lr_ratio
    return {
        "lr_scheduler": scheduler,
        "base_learning_rate": base_lr,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
        "min_learning_rate": min_learning_rate,
        "cosine_total_steps": int(total_steps),
    }

def scheduled_lr(schedule: dict, step: int) -> float:
    scheduler = schedule["lr_scheduler"]
    base_lr = float(schedule["base_learning_rate"])
    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"unsupported lr_scheduler: {scheduler}")
    warmup_steps = int(schedule["warmup_steps"])
    total_steps = max(int(schedule["cosine_total_steps"]), 1)
    min_lr = float(schedule["min_learning_rate"])
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))

def apply_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)

def graph_fields(args) -> dict:
    materialization = getattr(args, "graph_materialization", None)
    certificate = getattr(args, "graph_certificate", {}) or {}
    graph_artifact = getattr(args, "graph_artifact", {}) or {}
    fields = {
        "graph_generation_algorithm": graph_artifact.get("graph_generation_algorithm", ""),
        "canonical_graph_dir": "",
        "canonical_graph_artifact_path": "",
        "canonical_graph_artifact_sha256": "",
        "canonical_graph_seed": graph_artifact.get("graph_seed", ""),
        "canonical_graph_generation_algorithm": graph_artifact.get("graph_generation_algorithm", ""),
        "graph_generation_status": "",
        "graph_generation_attempts": "",
        "graph_artifact_path": getattr(args, "graph_artifact_path", ""),
        "graph_generation_path": "",
        "graph_certificate_path": "",
        "graph_artifact_sha256": "",
        "graph_artifact_sha256_matches_canonical": "",
        "graph_certificate_sha256": "",
        "rho_zigzag_bound": certificate.get("rho_zigzag_bound", certificate.get("rho_bound", "")),
        "rho_zigzag_certified": certificate.get("rho_zigzag_certified", certificate.get("certified", "")),
        "rho_zigzag_exact": certificate.get("rho_zigzag_exact", certificate.get("rho_exact", "")),
        "rot_g_is_bijection": certificate.get("rot_g_is_bijection", ""),
        "P_G_row_stochastic_error": certificate.get("P_G_row_stochastic_error", ""),
        "P_G_col_stochastic_error": certificate.get("P_G_col_stochastic_error", ""),
        "P_H_row_stochastic_error": certificate.get("P_H_row_stochastic_error", ""),
        "P_H_col_stochastic_error": certificate.get("P_H_col_stochastic_error", ""),
        "collision_count_mean": certificate.get("collision_count_mean", ""),
    }
    if materialization is not None:
        fields.update(materialization.as_dict())
        fields["graph_generation_algorithm"] = materialization.canonical_graph_generation_algorithm
        fields["graph_generation_status"] = materialization.graph_generation.get("status", "")
        fields["graph_generation_attempts"] = materialization.graph_generation.get("generation_attempts", "")
    return fields

def budget_fields(method: str, zigzag_budget: dict, random_budget: dict) -> dict:
    out = {
        "zigzag_actual_k_min_after_causal": zigzag_budget.get("actual_k_min_after_causal", ""),
        "zigzag_actual_k_mean_after_causal": zigzag_budget.get("actual_k_mean_after_causal", ""),
        "zigzag_actual_k_max_after_causal": zigzag_budget.get("actual_k_max_after_causal", ""),
        "zigzag_attention_pair_count_after_causal": zigzag_budget.get("attention_pair_count_after_causal", ""),
        "random_target_k_source": random_budget.get("random_target_k_source", ""),
        "random_actual_k_min_after_causal": random_budget.get("actual_k_min_after_causal", ""),
        "random_actual_k_mean_after_causal": random_budget.get("actual_k_mean_after_causal", ""),
        "random_actual_k_max_after_causal": random_budget.get("actual_k_max_after_causal", ""),
        "random_attention_pair_count_after_causal": random_budget.get("attention_pair_count_after_causal", ""),
        "random_k_alignment_error_mean": random_budget.get("random_k_alignment_error_mean", ""),
        "random_k_alignment_error_max": random_budget.get("random_k_alignment_error_max", ""),
        "random_alignment_mode": random_budget.get("random_alignment_mode", ""),
        "random_k_aligned_to_zigzag": random_budget.get("random_k_aligned_to_zigzag", ""),
    }
    if canonical_method(method) not in {"random_regular", "zigzag_certified", "zigzag_certified_cosine", "zigzag_boolean"}:
        out["random_k_aligned_to_zigzag"] = ""
    return out

def compact_budget_payload(payload: dict) -> dict:
    """Keep archived budget files compact while retaining aggregate diagnostics."""
    return {
        key: value
        for key, value in payload.items()
        if key != "per_query_k_after_causal"
    }

def train_method(
    method: str,
    args,
    device: torch.device,
    output_dir: Path,
    train_len: int | None = None,
    seed: int | None = None,
    eval_lengths: list[int] | None = None,
) -> dict:
    method = canonical_method(method)
    train_len = int(train_len if train_len is not None else args.seq_len)
    seed = int(seed if seed is not None else args.seed)
    eval_lengths = [int(v) for v in (eval_lengths if eval_lengths is not None else [train_len])]
    args = copy.copy(args)
    args.seq_len = train_len
    args.seed = seed
    if str(getattr(args, "version", "")).lower() == "v07" and getattr(args, "raw_config_snapshot", None):
        run_materialization = materialize_graph_artifact(
            args.raw_config_snapshot,
            output_dir,
            require=True,
        )
        args.graph_materialization = run_materialization
        args.graph_artifact = run_materialization.artifact
        args.graph_config = run_materialization.artifact
        args.graph_certificate = run_materialization.certificate
        args.graph_artifact_path = str(run_materialization.selected_graph_path)
        args.graph_id = str(run_materialization.artifact.get("graph_id", ""))
        args.graph_seed = run_materialization.artifact.get("graph_seed", "")
        args.config_snapshot.setdefault("attention", {})["graph_artifact"] = str(
            run_materialization.selected_graph_path
        )
        args.config_snapshot.setdefault("attention", {}).setdefault("runtime_graph", {}).update(
            run_materialization.as_dict()
        )
        args.config_snapshot.setdefault("graph", {})["runtime_graph_artifact_path"] = str(
            run_materialization.selected_graph_path
        )
        write_json(output_dir / "raw_config_snapshot.json", args.raw_config_snapshot)
        write_json(output_dir / "resolved_config_snapshot.json", args.config_snapshot)
        write_json(output_dir / "config_snapshot.json", args.config_snapshot)
    if method in {"zigzag_certified", "zigzag_certified_cosine"} and not bool(
        getattr(args, "graph_certificate", {}).get(
            "rho_zigzag_certified",
            getattr(args, "graph_certificate", {}).get("certified"),
        )
    ):
        raise ValueError("zigzag_certified requires a certified graph artifact")
    set_seed(args.seed)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    spec = TaskSpec(
        num_values=args.num_values,
        pad_token=args.pad_token,
        sep_token=args.sep_token,
        eos_token=args.eos_token,
    )
    attention_backend = resolve_attention_backend(args.attention_backend, method)
    train_T_raw, train_T = padded_copy_lengths(train_len, args.block_size)
    max_T = max(padded_copy_lengths(eval_len, args.block_size)[1] for eval_len in eval_lengths)
    zigzag_budget, random_budget, random_rows = budget_diagnostics(train_T, args)
    args.random_aligned_rows = random_rows
    write_json(output_dir / "zigzag_budget.json", compact_budget_payload(zigzag_budget))
    write_json(output_dir / "random_budget.json", compact_budget_payload(random_budget))
    train_artifacts = make_attention_artifacts(
        method, train_T, args, device, attention_backend
    )
    if getattr(args, "graph_certificate", None):
        write_json(output_dir / "graph_certificate.json", args.graph_certificate)
    model = Transformer(
        vocab_size=spec.vocab_size,
        output_size=spec.vocab_size,
        seq_len=max_T,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attention_backend=attention_backend,
        block_size=args.block_size,
    ).to(device)
    if args.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer: {args.optimizer}")
    schedule = method_schedule_config(args, method)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(schedule["base_learning_rate"]),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        cuda_sync(device)
    start = time.perf_counter()
    prev_log_time = start
    last_loss = None
    metrics_path = output_dir / "metrics.jsonl"
    method_metrics_path = output_dir / f"{method}_metrics.jsonl"
    metrics_rows: list[dict] = []
    with metrics_path.open("w", encoding="utf-8") as fp:
        for step in range(1, args.steps + 1):
            current_lr = scheduled_lr(schedule, step)
            apply_lr(opt, current_lr)
            batch = make_batch(
                args,
                spec,
                device,
                seq_len=train_len,
                batch_size=args.batch_size,
                batch_index=step,
                stream="train",
            )
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch.tokens,
                train_artifacts.mask,
                train_artifacts.local_valid,
                train_artifacts.neighbors,
                train_artifacts.valid_neighbors,
                train_artifacts.block_pair_index,
                train_artifacts.local_log_m,
                train_artifacts.neighbor_log_m,
            )
            loss, train_metrics = copy_loss_and_metrics(logits, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method} produced non-finite loss at step {step}")
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if (
                step == 1
                or step % args.log_every == 0
                or step % args.eval_every == 0
                or step == args.steps
            ):
                cuda_sync(device)
                now = time.perf_counter()
                elapsed_so_far = now - start
                seconds_since_prev_log = now - prev_log_time
                prev_log_time = now
                eval_metrics = evaluate(
                    model,
                    train_artifacts,
                    args,
                    spec,
                    device,
                    train_len,
                    stream=f"valid_N{train_len}",
                )
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "epoch": 0,
                    "method": method,
                    "seed": seed,
                    "elapsed_sec_total": elapsed_so_far,
                    "seconds_since_prev_log": seconds_since_prev_log,
                    "N_train": train_len,
                    "N_eval": train_len,
                    "train_loss": last_loss,
                    "train_token_accuracy": train_metrics["token_accuracy"],
                    "train_sequence_accuracy": train_metrics["sequence_accuracy"],
                    "train_eos_accuracy": train_metrics["eos_accuracy"],
                    "eval_loss": eval_metrics["loss"],
                    "eval_token_accuracy": eval_metrics["token_accuracy"],
                    "eval_sequence_accuracy": eval_metrics["sequence_accuracy"],
                    "eval_eos_accuracy": eval_metrics["eos_accuracy"],
                    "tokens_per_sec": step * args.batch_size * train_T / max(elapsed_so_far, 1e-12),
                    "learning_rate": current_lr,
                    "lr_scheduler": schedule["lr_scheduler"],
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
                    "effective_K_mean_after_causal": train_artifacts.metrics["effective_k_mean"],
                    "attention_pair_count_after_causal": train_artifacts.metrics["attention_pair_count"],
                }
                metrics_rows.append(row)
                fp.write(json.dumps(row) + "\n")
                fp.flush()
                print(json.dumps(row), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "step": step,
                        "training_seed": seed,
                        "graph_id": getattr(args, "graph_id", ""),
                        "config_snapshot": args.config_snapshot,
                        "rng_state": checkpoint_rng_state(device),
                    },
                    output_dir / f"checkpoint_step_{step:04d}.pt",
                )
    if torch.cuda.is_available():
        cuda_sync(device)
    elapsed = time.perf_counter() - start
    method_metrics_path.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    training_curves_path = output_dir / "training_curves.png"
    if args.plot_curves:
        plot_training_curves(metrics_rows, training_curves_path)

    base_result = {
        "method": method,
        "task": args.task,
        "attention_backend": attention_backend,
        "N_train": train_len,
        "T_raw": train_T_raw,
        "T": train_T,
        "block_size": args.block_size,
        "degree": args.degree,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batch_size": getattr(args, "eval_batch_size", args.batch_size),
        "final_train_loss": last_loss,
        "tokens_per_sec": args.steps * args.batch_size * train_T / elapsed,
        "elapsed_sec": elapsed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "mask": train_artifacts.metrics,
        "task_spec": asdict(spec),
        "metrics_path": str(metrics_path),
        "training_curves_path": str(training_curves_path) if args.plot_curves else "",
        "neighbor_shape": list(train_artifacts.neighbors.shape)
        if train_artifacts.neighbors is not None
        else None,
        "block_pair_shape": list(train_artifacts.block_pair_index.shape)
        if train_artifacts.block_pair_index is not None
        else None,
    }
    base_result.update(graph_fields(args))
    base_result.update(budget_fields(method, zigzag_budget, random_budget))
    shortcut_rows = compute_shortcut_stats(
        method=method,
        N=train_len,
        T=train_T,
        B=args.block_size,
        d=args.degree,
        seed=seed,
        layers=args.layers,
        graph_artifact=getattr(args, "graph_artifact", None),
    )
    write_csv(
        output_dir / "shortcut_diagnostics.csv",
        shortcut_rows,
        list(shortcut_rows[0].keys()) if shortcut_rows else [],
    )
    write_jsonl(output_dir / "shortcut_diagnostics.jsonl", shortcut_rows)
    shortcut_summary = next(
        (row for row in shortcut_rows if row.get("mask_scope") == "causal_effective"),
        shortcut_rows[0] if shortcut_rows else {},
    )
    base_result["shortcut_diagnostics"] = shortcut_rows
    records = []
    for eval_len in eval_lengths:
        eval_T_raw, eval_T = padded_copy_lengths(eval_len, args.block_size)
        eval_artifacts = make_attention_artifacts(
            method, eval_T, args, device, attention_backend
        )
        eval_metrics = evaluate(
            model,
            eval_artifacts,
            args,
            spec,
            device,
            eval_len,
            stream=f"eval_N{eval_len}",
        )
        graph_config = getattr(args, "graph_config", DEFAULT_GRAPH_CONFIG)
        g_type = graph_config.get("G", {}).get("type")
        h_type = graph_config.get("H", {}).get("type")
        certificate = getattr(args, "graph_certificate", {}) or {}
        cert_fields = method_certification_fields(
            method, certificate, getattr(args, "multiplicity_mode", "boolean")
        )
        run_id = f"train_N{train_len}_seed{seed}_{method}"
        record = {
            "version": getattr(args, "version", "v06"),
            "run_id": run_id,
            "status": "ok",
            "failure_reason": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "local_or_remote": getattr(args, "local_or_remote", "unknown"),
            "git_commit": getattr(args, "git_commit", ""),
            "config_path": getattr(args, "config_path", ""),
            "config_sha256": getattr(args, "config_sha256", ""),
            "command": getattr(args, "command", ""),
            "output_dir": str(getattr(args, "output_dir", "")),
            "log_path": getattr(args, "log_path", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "task": args.task,
            "data_mode": getattr(args, "data_mode", "online"),
            "num_values": args.num_values,
            "copy_mode": args.copy_mode,
            "sep_token": args.sep_token,
            "eos_token": args.eos_token,
            "pad_token": args.pad_token,
            "method": method,
            "graph_id": getattr(args, "graph_id", ""),
            "graph_seed": getattr(args, "graph_seed", ""),
            **graph_fields(args),
            "attention_backend": attention_backend,
            "N_total": getattr(args, "N_total", 2 * train_len + 2),
            "copy_source_length": getattr(args, "copy_source_length", train_len),
            "N_train": train_len,
            "N_eval": eval_len,
            "T_raw": eval_T_raw,
            "T": eval_T,
            "B": args.block_size,
            "d": args.degree,
            "G_type": g_type,
            "H_type": h_type,
            "causal": args.causal,
            "multiplicity_mode": getattr(args, "multiplicity_mode", "boolean"),
            "seed": seed,
            "architecture": args.architecture,
            "layers": args.layers,
            "d_model": args.d_model,
            "heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "base_learning_rate": schedule["base_learning_rate"],
            "lr_scheduler": schedule["lr_scheduler"],
            "warmup_ratio": schedule["warmup_ratio"],
            "warmup_steps": schedule["warmup_steps"],
            "min_lr_ratio": schedule["min_lr_ratio"],
            "min_learning_rate": schedule["min_learning_rate"],
            "cosine_total_steps": schedule["cosine_total_steps"],
            "weight_decay": getattr(args, "weight_decay", 0.0),
            "grad_clip_norm": getattr(args, "grad_clip_norm", 0.0),
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "checkpoint_every": getattr(args, "checkpoint_every", 0),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": getattr(args, "eval_batch_size", args.batch_size),
            "eval_batches": args.eval_batches,
            "raw_K": eval_artifacts.metrics["raw_k"],
            "unique_K_mean": eval_artifacts.metrics["unique_k_mean"],
            "effective_K_mean_after_causal": eval_artifacts.metrics["effective_k_mean"],
            "effective_K_min_after_causal": eval_artifacts.metrics["effective_k_min"],
            "effective_K_max_after_causal": eval_artifacts.metrics["effective_k_max"],
            "pre_causal_unique_K_mean": eval_artifacts.metrics["pre_causal_unique_k_mean"],
            "pre_causal_pair_count": eval_artifacts.metrics["pre_causal_pair_count"],
            "duplicate_rate": eval_artifacts.metrics["duplicate_rate"],
            "self_loop_rate": eval_artifacts.metrics["self_loop_rate"],
            "attention_pair_count_after_causal": eval_artifacts.metrics["attention_pair_count"],
            **budget_fields(method, zigzag_budget, random_budget),
            "lambda_G": certificate.get("lambda_G", ""),
            "mu_H": certificate.get("mu_H", ""),
            "rho_bound": certificate.get("rho_bound", ""),
            "rho_zigzag_bound": certificate.get("rho_zigzag_bound", certificate.get("rho_bound", "")),
            "rho_zigzag_certified": certificate.get("rho_zigzag_certified", certificate.get("certified", "")),
            "rho_exact": certificate.get("rho_exact", ""),
            "rho_zigzag_exact": certificate.get("rho_zigzag_exact", certificate.get("rho_exact", "")),
            "certified": cert_fields["certified"],
            "graph_certified": cert_fields["graph_certified"],
            "implementation_certified": cert_fields["implementation_certified"],
            "theory_aligned_method": cert_fields["theory_aligned_method"],
            "remote_local_overlap_mean": certificate.get("remote_local_overlap_mean", ""),
            "target_in_1hop_rate": shortcut_summary.get("target_in_1hop_rate", ""),
            "target_in_2hop_rate": shortcut_summary.get("target_in_2hop_rate", ""),
            "target_in_Lhop_rate": shortcut_summary.get("target_in_Lhop_rate", ""),
            "average_shortest_path": shortcut_summary.get("average_shortest_path", ""),
            "unreachable_rate": shortcut_summary.get("unreachable_rate", ""),
            "final_train_loss": last_loss,
            "eval_loss": eval_metrics["loss"],
            "eval_token_accuracy": eval_metrics["token_accuracy"],
            "eval_sequence_accuracy": eval_metrics["sequence_accuracy"],
            "eval_eos_accuracy": eval_metrics["eos_accuracy"],
            "training_curves_path": str(training_curves_path) if args.plot_curves else "",
            "tokens_per_sec": args.steps * args.batch_size * train_T / elapsed,
            "elapsed_sec": elapsed,
            "total_wall_time_sec": elapsed,
            "train_wall_time_sec": elapsed,
            "eval_wall_time_sec": 0.0,
            "data_prep_wall_time_sec": 0.0,
            "peak_allocated_gb": base_result["peak_allocated_gb"],
            "peak_reserved_gb": base_result["peak_reserved_gb"],
            "artifact_dir": str(output_dir),
            "metrics_path": str(metrics_path),
            "summary_path": str(output_dir / "summary.json"),
            "raw_config_snapshot_path": str(output_dir / "raw_config_snapshot.json"),
            "resolved_config_snapshot_path": str(output_dir / "resolved_config_snapshot.json"),
            "neighbor_shape": base_result["neighbor_shape"],
            "block_pair_shape": base_result["block_pair_shape"],
        }
        records.append(record)
    base_result["evals"] = records
    if records:
        base_result["final_valid_loss"] = records[0]["eval_loss"]
        base_result["final_valid_token_accuracy"] = records[0]["eval_token_accuracy"]
    write_csv(output_dir / "results.csv", records, RESULT_FIELDS)
    write_jsonl(output_dir / "results.jsonl", records)
    write_json(
        output_dir / "summary.json",
        {
            "status": "ok",
            "run_id": f"train_N{train_len}_seed{seed}_{method}",
            "config": getattr(args, "config_snapshot", {}),
            "result": base_result,
            "results": records,
        },
    )
    return base_result
