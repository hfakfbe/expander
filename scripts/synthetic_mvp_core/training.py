from __future__ import annotations

import copy
import gc
import json
import os
import random
import socket
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graph_diagnostics import compute_shortcut_stats
from graph_structures import DEFAULT_GRAPH_CONFIG, canonical_method

from .artifacts import (
    RESULT_FIELDS,
    AttentionArtifacts,
    make_attention_artifacts,
    method_certification_fields,
    resolve_attention_backend,
)
from .common import set_seed
from .config import serialize_args
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
        for batch_idx in range(args.eval_batches):
            batch = make_batch(
                args,
                spec,
                device,
                seq_len=N,
                batch_size=args.batch_size,
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
        "loss": total_loss / args.eval_batches,
        "token_accuracy": total_token_correct / total_tokens,
        "sequence_accuracy": total_sequence_correct / total_sequences,
        "eos_accuracy": total_eos_correct / total_sequences,
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
    if method == "zigzag_certified" and not bool(getattr(args, "graph_certificate", {}).get("certified")):
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        cuda_sync(device)
    start = time.perf_counter()
    last_loss = None
    metrics_path = output_dir / "metrics.jsonl"
    method_metrics_path = output_dir / f"{method}_metrics.jsonl"
    metrics_rows: list[dict] = []
    with metrics_path.open("w", encoding="utf-8") as fp:
        for step in range(1, args.steps + 1):
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
                elapsed_so_far = time.perf_counter() - start
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
                    "step": step,
                    "method": method,
                    "seed": seed,
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
            "attention_backend": attention_backend,
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
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "steps": args.steps,
            "batch_size": args.batch_size,
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
            "lambda_G": certificate.get("lambda_G", ""),
            "mu_H": certificate.get("mu_H", ""),
            "rho_bound": certificate.get("rho_bound", ""),
            "rho_exact": certificate.get("rho_exact", ""),
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
            "peak_allocated_gb": base_result["peak_allocated_gb"],
            "peak_reserved_gb": base_result["peak_reserved_gb"],
            "artifact_dir": str(output_dir),
            "metrics_path": str(metrics_path),
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
            "config": serialize_args(args),
            "result": base_result,
            "results": records,
        },
    )
    return base_result
