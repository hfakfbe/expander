from __future__ import annotations

import copy
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from graph_diagnostics import CERTIFICATE_FIELDS
from graph_structures import DEFAULT_GRAPH_CONFIG

from .artifacts import RESULT_FIELDS, method_certification_fields
from .common import set_seed
from .config import (
    apply_cli_overrides,
    build_runtime_args,
    load_config,
    parse_args,
    serialize_args,
    write_command_script,
)
from .data import padded_copy_lengths
from .io_utils import write_csv, write_json, write_jsonl
from .mask_tests import run_mask_tests
from .training import train_method


def failure_record(args, run_id: str, train_len: int, seed: int, method: str, error: Exception, run_dir: Path) -> dict:
    graph_config = getattr(args, "graph_config", DEFAULT_GRAPH_CONFIG)
    certificate = getattr(args, "graph_certificate", {}) or {}
    cert_fields = method_certification_fields(
        method, certificate, getattr(args, "multiplicity_mode", "boolean")
    )
    T_raw, T = padded_copy_lengths(train_len, args.block_size)
    graph_materialization = getattr(args, "graph_materialization", None)
    graph_extra = graph_materialization.as_dict() if graph_materialization is not None else {}
    return {
        "version": getattr(args, "version", "v06"),
        "run_id": run_id,
        "status": "failed",
        "failure_reason": repr(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "local_or_remote": args.local_or_remote,
        "git_commit": args.git_commit,
        "config_path": args.config_path,
        "config_sha256": args.config_sha256,
        "command": args.command,
        "output_dir": str(args.output_dir),
        "log_path": args.log_path,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "task": args.task,
        "data_mode": args.data_mode,
        "num_values": args.num_values,
        "copy_mode": args.copy_mode,
        "sep_token": args.sep_token,
        "eos_token": args.eos_token,
        "pad_token": args.pad_token,
        "method": method,
        "graph_id": getattr(args, "graph_id", ""),
        "graph_seed": getattr(args, "graph_seed", ""),
        "graph_generation_algorithm": getattr(args, "graph_artifact", {}).get("graph_generation_algorithm", ""),
        **graph_extra,
        "attention_backend": args.attention_backend,
        "N_total": getattr(args, "N_total", 2 * train_len + 2),
        "copy_source_length": getattr(args, "copy_source_length", train_len),
        "N_train": train_len,
        "N_eval": "",
        "T_raw": T_raw,
        "T": T,
        "B": args.block_size,
        "d": args.degree,
        "G_type": graph_config.get("G", {}).get("type"),
        "H_type": graph_config.get("H", {}).get("type"),
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
        "eval_every": getattr(args, "eval_every", ""),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "raw_K": "",
        "unique_K_mean": "",
        "effective_K_mean_after_causal": "",
        "effective_K_min_after_causal": "",
        "effective_K_max_after_causal": "",
        "pre_causal_unique_K_mean": "",
        "pre_causal_pair_count": "",
        "duplicate_rate": "",
        "self_loop_rate": "",
        "attention_pair_count_after_causal": "",
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
        "target_in_1hop_rate": "",
        "target_in_2hop_rate": "",
        "target_in_Lhop_rate": "",
        "average_shortest_path": "",
        "unreachable_rate": "",
        "final_train_loss": "",
        "eval_loss": "",
        "eval_token_accuracy": "",
        "eval_sequence_accuracy": "",
        "eval_eos_accuracy": "",
        "training_curves_path": "",
        "tokens_per_sec": "",
        "elapsed_sec": "",
        "total_wall_time_sec": "",
        "train_wall_time_sec": "",
        "eval_wall_time_sec": "",
        "data_prep_wall_time_sec": "",
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
        "artifact_dir": str(run_dir),
        "metrics_path": "",
        "summary_path": str(run_dir / "summary.json"),
        "raw_config_snapshot_path": str(run_dir / "raw_config_snapshot.json"),
        "resolved_config_snapshot_path": str(run_dir / "resolved_config_snapshot.json"),
        "neighbor_shape": "",
        "block_pair_shape": "",
    }

def main() -> None:
    started = time.perf_counter()
    cli = parse_args()
    config, config_path, config_sha = load_config(cli.config)
    config = apply_cli_overrides(config, cli)
    args = build_runtime_args(config, cli, config_path, config_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    set_seed(args.seed)
    write_json(args.output_dir / "config_snapshot.json", args.config_snapshot)
    write_json(args.output_dir / "raw_config_snapshot.json", args.raw_config_snapshot)
    write_json(args.output_dir / "resolved_config_snapshot.json", args.config_snapshot)
    write_command_script(args.output_dir / "command.sh", args.command)

    test_results = []
    if not args.skip_tests:
        test_results = run_mask_tests(device)
        write_json(args.output_dir / "mask_tests.json", test_results)
        print(json.dumps({"mask_tests": "ok", "cases": len(test_results)}), flush=True)

    all_records: list[dict] = []
    method_results: list[dict] = []
    metrics_lines: list[str] = []
    shortcut_rows_all: list[dict] = []
    for train_len in args.train_lengths:
        for seed in args.seeds:
            for method in args.methods:
                run_id = f"train_N{train_len}_seed{seed}_{method}"
                run_dir = args.output_dir / run_id
                existing_summary = run_dir / "summary.json"
                if existing_summary.exists():
                    try:
                        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
                        existing_rows = existing.get("results", [])
                        curve_path = run_dir / "training_curves.png"
                        if (
                            existing.get("status") == "ok"
                            and len(existing_rows) >= len(args.eval_lengths)
                            and all(row.get("status") == "ok" for row in existing_rows)
                            and all("eval_token_accuracy" in row for row in existing_rows)
                            and curve_path.exists()
                        ):
                            all_records.extend(existing_rows)
                            result = existing.get("result")
                            if result is not None:
                                method_results.append(result)
                            metrics_path = run_dir / "metrics.jsonl"
                            if metrics_path.exists():
                                metrics_lines.extend(metrics_path.read_text(encoding="utf-8").splitlines())
                            shortcut_path = run_dir / "shortcut_diagnostics.jsonl"
                            if shortcut_path.exists():
                                shortcut_rows_all.extend(
                                    json.loads(line)
                                    for line in shortcut_path.read_text(encoding="utf-8").splitlines()
                                    if line.strip()
                                )
                            print(json.dumps({"skipped_completed": run_id}), flush=True)
                            continue
                    except Exception:
                        pass
                run_dir.mkdir(parents=True, exist_ok=True)
                run_args = copy.copy(args)
                run_args.output_dir = run_dir
                write_json(run_dir / "config_snapshot.json", args.config_snapshot)
                write_json(run_dir / "raw_config_snapshot.json", args.raw_config_snapshot)
                write_json(run_dir / "resolved_config_snapshot.json", args.config_snapshot)
                write_command_script(run_dir / "command.sh", args.command)
                try:
                    result = train_method(
                        method,
                        run_args,
                        device,
                        run_dir,
                        train_len=train_len,
                        seed=seed,
                        eval_lengths=args.eval_lengths,
                    )
                    method_results.append(result)
                    all_records.extend(result["evals"])
                    shortcut_rows_all.extend(result.get("shortcut_diagnostics", []))
                    metrics_lines.extend((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines())
                    print(json.dumps({"completed": run_id, "evals": result["evals"]}, indent=2), flush=True)
                except Exception as exc:
                    (run_dir / "error.log").write_text(repr(exc) + "\n", encoding="utf-8")
                    failed = failure_record(args, run_id, train_len, seed, method, exc, run_dir)
                    all_records.append(failed)
                    write_csv(run_dir / "results.csv", [failed], RESULT_FIELDS)
                    write_jsonl(run_dir / "results.jsonl", [failed])
                    write_json(
                        run_dir / "summary.json",
                        {
                            "status": "failed",
                            "run_id": run_id,
                            "config": getattr(args, "config_snapshot", {}),
                            "failure_reason": repr(exc),
                            "results": [failed],
                        },
                    )
                    print(json.dumps({"failed": run_id, "error": repr(exc)}), flush=True)

    if metrics_lines:
        (args.output_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "results.csv", all_records, RESULT_FIELDS)
    write_jsonl(args.output_dir / "results.jsonl", all_records)
    write_csv(args.output_dir / "phase5_results.csv", all_records, RESULT_FIELDS)
    write_jsonl(args.output_dir / "phase5_results.jsonl", all_records)
    write_csv(args.output_dir / "phase3_results.csv", all_records, RESULT_FIELDS)
    write_jsonl(args.output_dir / "phase3_results.jsonl", all_records)
    if shortcut_rows_all:
        write_csv(
            args.output_dir / "shortcut_diagnostics.csv",
            shortcut_rows_all,
            list(shortcut_rows_all[0].keys()),
        )
        write_jsonl(args.output_dir / "shortcut_diagnostics.jsonl", shortcut_rows_all)
    if getattr(args, "graph_certificate", None):
        write_csv(args.output_dir / "graph_diagnostics.csv", [args.graph_certificate], CERTIFICATE_FIELDS)
        write_json(args.output_dir / "graph_certificate.json", args.graph_certificate)
    status = "ok" if all(row.get("status") == "ok" for row in all_records) else "failed"
    write_json(
        args.output_dir / "summary.json",
        {
            "status": status,
            "config": getattr(args, "config_snapshot", {}),
            "raw_config_snapshot_path": str(args.output_dir / "raw_config_snapshot.json"),
            "resolved_config_snapshot_path": str(args.output_dir / "resolved_config_snapshot.json"),
            "total_wall_time_sec": time.perf_counter() - started,
            "mask_test_cases": len(test_results),
            "results": all_records,
            "method_results": method_results,
        },
    )
    if status != "ok":
        raise SystemExit(1)
