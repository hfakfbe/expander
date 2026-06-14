from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from graph_structures import canonical_method
from synthetic_mvp_core.artifacts import (
    RESULT_FIELDS,
    budget_diagnostics,
    make_attention_artifacts,
    method_certification_fields,
    resolve_attention_backend,
)
from synthetic_mvp_core.io_utils import plot_training_curves, write_csv, write_json, write_jsonl
from synthetic_mvp_core.training import (
    apply_lr,
    budget_fields,
    compact_budget_payload,
    method_schedule_config,
    scheduled_lr,
)
from v07_artifacts import file_sha256, git_commit, materialize_graph_artifact
from wikitext2_utils import (
    build_resolved_config_snapshot,
    build_model_and_artifacts,
    build_runtime,
    copy_phase4_artifacts,
    load_phase4_tokenizer,
    load_tokenized_blocks,
    lm_loss_and_metrics,
    make_lm_batch,
    read_json,
    run_eval_batches,
    write_command,
)


WIKITEXT_EXTRA_FIELDS = [
    "dataset",
    "dataset_source",
    "dataset_revision_or_hash",
    "dataset_cache_or_local_path",
    "wikitext_data_phase_dir",
    "data_readiness_path",
    "data_readiness_sha256",
    "tokenization_summary_path",
    "tokenization_summary_sha256",
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
    "tokenized_train_path",
    "tokenized_train_sha256",
    "tokenized_test_path",
    "tokenized_test_sha256",
    "train_steps",
    "train_epoch_count",
    "train_token_count",
    "train_block_count",
    "test_token_count",
    "test_block_count",
    "final_train_loss",
    "test_loss",
    "test_perplexity",
    "train_tokens_per_sec",
    "test_tokens_per_sec",
]

for field in WIKITEXT_EXTRA_FIELDS:
    if field not in RESULT_FIELDS:
        RESULT_FIELDS.append(field)


def finite_loss(value: float, method: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"non-finite loss for {method}")


def data_context(args, tokenizer) -> dict:
    readiness = read_json(args.data_readiness_path)
    tokenization = read_json(args.tokenization_summary_path)
    tokenizer_sha = file_sha256(args.tokenizer_path)
    train_sha = file_sha256(args.tokenized_train_path)
    test_sha = file_sha256(args.tokenized_test_path)
    if args.require_tokenizer_sha256_match and args.expected_tokenizer_sha256 and tokenizer_sha != args.expected_tokenizer_sha256:
        raise ValueError("tokenizer sha256 mismatch")
    if args.require_data_sha256_match and args.expected_tokenized_train_sha256 and train_sha != args.expected_tokenized_train_sha256:
        raise ValueError("tokenized train sha256 mismatch")
    if args.require_data_sha256_match and args.expected_tokenized_test_sha256 and test_sha != args.expected_tokenized_test_sha256:
        raise ValueError("tokenized test sha256 mismatch")
    tok_cfg = read_json(args.tokenizer_source_dir / "tokenizer_config.json")
    return {
        "dataset": readiness.get("dataset", "wikitext-103-raw-v1"),
        "dataset_source": readiness.get("dataset_source", ""),
        "dataset_revision_or_hash": readiness.get("dataset_revision_or_hash", ""),
        "dataset_cache_or_local_path": readiness.get("dataset_cache_or_local_path", ""),
        "wikitext_data_phase_dir": str(args.data_phase_dir),
        "data_readiness_path": str(args.data_readiness_path),
        "data_readiness_sha256": file_sha256(args.data_readiness_path),
        "tokenization_summary_path": str(args.tokenization_summary_path),
        "tokenization_summary_sha256": file_sha256(args.tokenization_summary_path),
        "tokenizer": tokenizer.name,
        "tokenizer_algorithm": tok_cfg.get("tokenizer_algorithm", "byte_level_bpe"),
        "tokenizer_train_split": tok_cfg.get("tokenizer_train_split", "train"),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_min_frequency": tok_cfg.get("tokenizer_min_frequency", ""),
        "tokenizer_special_tokens": ",".join(tok_cfg.get("special_tokens", [])),
        "pad_token": tok_cfg.get("pad_token", "<pad>"),
        "eos_token": tok_cfg.get("eos_token", "<eos>"),
        "unk_token": tok_cfg.get("unk_token", "<unk>"),
        "vocab_size": tokenizer.vocab_size,
        "train_nonempty_rows": readiness.get("train_nonempty_rows", ""),
        "test_nonempty_rows": readiness.get("test_nonempty_rows", ""),
        "tokenized_train_path": str(args.tokenized_train_path),
        "tokenized_train_sha256": train_sha,
        "tokenized_test_path": str(args.tokenized_test_path),
        "tokenized_test_sha256": test_sha,
        "train_token_count": tokenization.get("train_token_count", ""),
        "train_block_count": tokenization.get("train_block_count", ""),
        "test_token_count": tokenization.get("test_token_count", ""),
        "test_block_count": tokenization.get("test_block_count", ""),
    }


def graph_context(args) -> dict:
    materialization = getattr(args, "graph_materialization", None)
    graph = args.graph_artifact
    cert = args.graph_certificate
    out = {
        "graph_generation_algorithm": graph.get("graph_generation_algorithm", ""),
        "graph_artifact_path": args.graph_artifact_path,
        "rho_zigzag_bound": cert.get("rho_zigzag_bound", cert.get("rho_bound", "")),
        "rho_zigzag_certified": cert.get("rho_zigzag_certified", cert.get("certified", "")),
        "rho_zigzag_exact": cert.get("rho_zigzag_exact", cert.get("rho_exact", "")),
        "rot_g_is_bijection": cert.get("rot_g_is_bijection", ""),
        "P_G_row_stochastic_error": cert.get("P_G_row_stochastic_error", ""),
        "P_G_col_stochastic_error": cert.get("P_G_col_stochastic_error", ""),
        "P_H_row_stochastic_error": cert.get("P_H_row_stochastic_error", ""),
        "P_H_col_stochastic_error": cert.get("P_H_col_stochastic_error", ""),
        "collision_count_mean": cert.get("collision_count_mean", ""),
    }
    if materialization is not None:
        out.update(materialization.as_dict())
        out["graph_generation_algorithm"] = materialization.canonical_graph_generation_algorithm
        out["graph_generation_status"] = materialization.graph_generation.get("status", "")
        out["graph_generation_attempts"] = materialization.graph_generation.get("generation_attempts", "")
    return out


def eval_batches_value(args, blocks: torch.Tensor) -> int:
    if args.eval_batches == "all":
        return max(1, math.ceil(len(blocks) / max(args.eval_batch_size, 1)))
    return int(args.eval_batches)


def tensor_shape(value) -> str:
    if value is None:
        return ""
    return "x".join(str(dim) for dim in value.shape)


def train_one_method(
    model,
    artifacts,
    train_blocks,
    args,
    tokenizer,
    method: str,
    metrics_path: Path,
) -> tuple[float, int, list[dict], dict]:
    if args.steps <= 0 and args.epochs <= 0:
        schedule = method_schedule_config(args, method)
        return float("nan"), 0, [], schedule
    model.train()
    rows: list[dict] = []
    total_steps = int(args.steps or (args.epochs * max(1, len(train_blocks) // args.batch_size)))
    total_steps = max(total_steps, int(getattr(args, "max_train_batches", 0) or 0))
    if args.steps <= 0 and args.epochs > 0:
        total_steps = args.epochs * max(1, len(train_blocks) // args.effective_batch_size)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    schedule_args = copy.copy(args)
    schedule_args.steps = total_steps
    schedule = method_schedule_config(schedule_args, method)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(schedule["base_learning_rate"]),
        weight_decay=args.weight_decay,
    )
    last_loss = float("nan")
    started = time.perf_counter()
    prev = started
    with metrics_path.open("w", encoding="utf-8") as fp:
        for step in range(1, total_steps + 1):
            current_lr = scheduled_lr(schedule, step)
            apply_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for accum_index in range(args.gradient_accumulation_steps):
                microbatch_index = (step - 1) * args.gradient_accumulation_steps + accum_index + 1
                batch = make_lm_batch(
                    train_blocks,
                    args.batch_size,
                    microbatch_index,
                    args.device,
                    args.sequence_length,
                    args.T,
                    args.seed,
                    f"train_{method}",
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
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
                loss, _ = lm_loss_and_metrics(logits, batch)
                finite_loss(float(loss.item()), method)
                accumulated_loss += float(loss.item())
                (loss / args.gradient_accumulation_steps).backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            last_loss = accumulated_loss / args.gradient_accumulation_steps
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                now = time.perf_counter()
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "epoch": 0,
                    "method": method,
                    "train_loss": last_loss,
                    "running_train_perplexity": math.exp(min(last_loss, 20.0)),
                    "learning_rate": current_lr,
                    "lr_scheduler": schedule["lr_scheduler"],
                    "elapsed_sec_total": now - started,
                    "seconds_since_prev_log": now - prev,
                    "tokens_per_sec": step * args.effective_batch_size * args.sequence_length / max(now - started, 1e-9),
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if args.device.type == "cuda" else 0.0,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if args.device.type == "cuda" else 0.0,
                }
                prev = now
                rows.append(row)
                fp.write(json.dumps(row, sort_keys=True) + "\n")
                fp.flush()
    return last_loss, total_steps, rows, schedule


def run_eval(config: dict, output_dir: Path, device: torch.device, log_path: str = "") -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_runtime(config, output_dir, device)
    tokenizer = load_phase4_tokenizer(args.tokenizer_source_dir)
    manifest = copy_phase4_artifacts(args, output_dir)
    write_json(output_dir / "raw_config_snapshot.json", args.raw_config_snapshot)
    write_json(output_dir / "resolved_config_snapshot.json", args.config_snapshot)
    train_blocks = load_tokenized_blocks(args.tokenized_train_path)
    test_blocks = load_tokenized_blocks(args.tokenized_test_path)
    data_ctx = data_context(args, tokenizer)
    certificate = args.graph_certificate
    graph = args.graph_artifact
    records: list[dict] = []
    metrics_rows: list[dict] = []

    max_train_batches = config.get("smoke", {}).get("max_train_batches")
    if max_train_batches is not None:
        args.steps = int(max_train_batches)

    for method in args.methods:
        method = canonical_method(method)
        run_dir = output_dir / f"seed{args.seed}_{method}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_materialization = materialize_graph_artifact(config, run_dir, require=True)
        args.graph_materialization = run_materialization
        args.graph_artifact = run_materialization.artifact
        args.graph_config = run_materialization.artifact
        args.graph_certificate = run_materialization.certificate
        args.graph_artifact_path = str(run_materialization.selected_graph_path)
        run_config = build_resolved_config_snapshot(
            config,
            run_materialization,
            str(run_materialization.selected_graph_path),
        )
        copy_phase4_artifacts(args, run_dir)
        write_json(run_dir / "raw_config_snapshot.json", args.raw_config_snapshot)
        write_json(run_dir / "resolved_config_snapshot.json", run_config)
        write_json(run_dir / "config_snapshot.json", run_config)
        write_command(run_dir / "command.sh")
        method_started = time.perf_counter()
        try:
            torch.manual_seed(args.seed)
            zigzag_budget, random_budget, random_rows = budget_diagnostics(args.T, args)
            args.random_aligned_rows = random_rows
            write_json(run_dir / "zigzag_budget.json", compact_budget_payload(zigzag_budget))
            write_json(run_dir / "random_budget.json", compact_budget_payload(random_budget))
            model, artifacts, backend = build_model_and_artifacts(args, method, tokenizer, device)
            metrics_path = run_dir / "metrics.jsonl"
            final_train_loss, train_steps, rows, schedule = train_one_method(
                model, artifacts, train_blocks, args, tokenizer, method, metrics_path
            )
            eval_args = copy.copy(args)
            eval_args.batch_size = args.eval_batch_size
            eval_args.eval_batches = eval_batches_value(args, test_blocks)
            test_started = time.perf_counter()
            test = run_eval_batches(model, artifacts, test_blocks, eval_args, tokenizer, device, "test")
            test_wall = time.perf_counter() - test_started
            finite_loss(float(test["test_loss"]), method)
            if rows:
                rows[-1]["test_loss"] = test["test_loss"]
                rows[-1]["test_perplexity"] = test["test_perplexity"]
                metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            if rows:
                plot_training_curves(rows, run_dir / "training_curves.png")
            elapsed = max(time.perf_counter() - method_started, 1e-9)
            certificate = args.graph_certificate
            graph = args.graph_artifact
            cert_fields = method_certification_fields(method, certificate, args.multiplicity_mode)
            record = {
                "version": args.version,
                "task": "wikitext",
                "run_id": f"seed{args.seed}_{method}",
                "status": "ok",
                "failure_reason": "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "local_or_remote": "remote" if str(Path.cwd()).startswith("/home/huiwei") else "local",
                "command": shlex.join([sys.executable, *sys.argv]),
                "log_path": log_path,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "method": method,
                "attention_backend": backend,
                "seed": args.seed,
                "N_total": args.sequence_length,
                "B": args.block_size,
                "q": graph.get("q", ""),
                "d": args.degree,
                "causal": args.causal,
                "graph_id": args.graph_id,
                "graph_seed": args.graph_seed,
                **graph_context(args),
                "G_type": graph.get("G", {}).get("type", ""),
                "H_type": graph.get("H", {}).get("type", ""),
                "allow_multiedges": graph.get("allow_multiedges", True),
                "multiplicity_mode": args.multiplicity_mode,
                "lambda_G": certificate.get("lambda_G", ""),
                "mu_H": certificate.get("mu_H", ""),
                "graph_certified": cert_fields["graph_certified"],
                "implementation_certified": cert_fields["implementation_certified"],
                "theory_aligned_method": cert_fields["theory_aligned_method"],
                "raw_K": artifacts.metrics.get("raw_k", ""),
                "unique_K_mean": artifacts.metrics.get("unique_k_mean", ""),
                "effective_K_mean_after_causal": artifacts.metrics.get("effective_k_mean", ""),
                "effective_K_min_after_causal": artifacts.metrics.get("effective_k_min", ""),
                "effective_K_max_after_causal": artifacts.metrics.get("effective_k_max", ""),
                "pre_causal_unique_K_mean": artifacts.metrics.get("pre_causal_unique_k_mean", ""),
                "pre_causal_pair_count": artifacts.metrics.get("pre_causal_pair_count", ""),
                "duplicate_rate": artifacts.metrics.get("duplicate_rate", ""),
                "self_loop_rate": artifacts.metrics.get("self_loop_rate", ""),
                "remote_local_overlap_mean": certificate.get("remote_local_overlap_mean", ""),
                "attention_pair_count_after_causal": artifacts.metrics.get("attention_pair_count", ""),
                **budget_fields(method, zigzag_budget, random_budget),
                "layers": args.layers,
                "d_model": args.d_model,
                "heads": args.heads,
                "ffn_dim": args.ffn_dim,
                "dropout": args.dropout,
                "optimizer": "adamw",
                "steps": train_steps,
                "train_steps": train_steps,
                "train_epochs": args.epochs,
                "train_epoch_count": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.effective_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "eval_batches": eval_args.eval_batches,
                "learning_rate": schedule["base_learning_rate"],
                "base_learning_rate": schedule["base_learning_rate"],
                "lr_scheduler": schedule["lr_scheduler"],
                "warmup_ratio": schedule["warmup_ratio"],
                "warmup_steps": schedule["warmup_steps"],
                "min_lr_ratio": schedule["min_lr_ratio"],
                "min_learning_rate": schedule["min_learning_rate"],
                "cosine_total_steps": schedule["cosine_total_steps"],
                "weight_decay": args.weight_decay,
                "grad_clip_norm": args.grad_clip_norm,
                "log_every": args.log_every,
                "eval_every": args.eval_every,
                "checkpoint_every": 0,
                "training_curves_path": str(run_dir / "training_curves.png"),
                "total_wall_time_sec": elapsed,
                "train_wall_time_sec": elapsed - test_wall,
                "eval_wall_time_sec": test_wall,
                "data_prep_wall_time_sec": 0.0,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0,
                "artifact_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "summary_path": str(run_dir / "summary.json"),
                "raw_config_snapshot_path": str(run_dir / "raw_config_snapshot.json"),
                "resolved_config_snapshot_path": str(run_dir / "resolved_config_snapshot.json"),
                "neighbor_shape": tensor_shape(artifacts.neighbors),
                "block_pair_shape": tensor_shape(artifacts.block_pair_index),
                "git_commit": git_commit(Path.cwd()),
                "config_sha256": "",
                **data_ctx,
                "final_train_loss": final_train_loss,
                "test_loss": test["test_loss"],
                "test_perplexity": test["test_perplexity"],
                "train_tokens_per_sec": train_steps
                * args.effective_batch_size
                * args.sequence_length
                / max(elapsed - test_wall, 1e-9),
                "test_tokens_per_sec": test.get("test_tokens_per_sec", ""),
            }
            write_json(run_dir / "summary.json", record)
            records.append(record)
            metrics_rows.extend(rows)
        except Exception as exc:
            record = {
                "version": args.version,
                "task": "wikitext",
                "run_id": f"seed{args.seed}_{method}",
                "status": "failed",
                "failure_reason": repr(exc),
                "dataset": "wikitext-103-raw-v1",
                "method": method,
                "seed": args.seed,
                "artifact_dir": str(run_dir),
            }
            (run_dir / "error.log").write_text(repr(exc) + "\n", encoding="utf-8")
            write_json(run_dir / "summary.json", record)
            records.append(record)

    write_csv(output_dir / "results.csv", records, RESULT_FIELDS)
    write_jsonl(output_dir / "results.jsonl", records)
    write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
    summary = {
        "status": "ok" if all(row.get("status") == "ok" for row in records) else "failed",
        "pipeline_only": False,
        "phase4_manifest": manifest,
        "total_wall_time_sec": time.perf_counter() - started,
        "results": records,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--local-or-remote", default="")
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method override for parallel method-only runs. Accepts space or comma separated names.",
    )
    return parser.parse_args()


def parse_method_override(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    methods: list[str] = []
    for value in values:
        methods.extend(method.strip() for method in value.split(",") if method.strip())
    return methods or None


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    methods = parse_method_override(args.methods)
    if methods is not None:
        config = copy.deepcopy(config)
        config.setdefault("attention", {})["methods"] = methods
    output_dir = args.output_dir or Path(config["output"]["root"])
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "raw_config_snapshot.json", config)
    write_command(output_dir / "command.sh")
    summary = run_eval(config, output_dir, device, log_path=args.log_path)
    print(json.dumps({"status": summary["status"], "pipeline_only": summary["pipeline_only"]}), flush=True)
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
