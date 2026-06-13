from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

from wikitext2_utils import (
    ByteTokenizer,
    build_blocks,
    build_model_and_artifacts,
    build_runtime,
    lm_loss_and_metrics,
    make_lm_batch,
    method_certification_fields,
    read_json,
    run_eval_batches,
    write_command,
    write_csv,
    write_json,
    write_jsonl,
)


RESULT_FIELDS = [
    "version",
    "run_id",
    "status",
    "failure_reason",
    "dataset",
    "dataset_source",
    "dataset_revision_or_hash",
    "tokenizer",
    "vocab_size",
    "sequence_length",
    "T",
    "method",
    "seed",
    "graph_id",
    "graph_seed",
    "B",
    "d",
    "q",
    "G_type",
    "H_type",
    "causal",
    "multiplicity_mode",
    "graph_certified",
    "implementation_certified",
    "theory_aligned_method",
    "architecture",
    "layers",
    "d_model",
    "heads",
    "ffn_dim",
    "steps",
    "batch_size",
    "eval_batches",
    "learning_rate",
    "log_every",
    "eval_every",
    "train_nonempty_rows",
    "validation_nonempty_rows",
    "test_nonempty_rows",
    "validation_loss",
    "validation_perplexity",
    "test_loss",
    "test_perplexity",
    "tokens_per_sec",
    "elapsed_sec",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "pipeline_only",
    "artifact_dir",
    "git_commit",
    "config_sha256",
]


def finite_loss(value: float, method: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"non-finite loss for {method}")


def train_steps(model, artifacts, train_blocks, args, device) -> None:
    if args.steps <= 0:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for step in range(1, args.steps + 1):
        batch = make_lm_batch(
            train_blocks,
            args.batch_size,
            step,
            device,
            args.sequence_length,
            args.T,
            args.seed,
            "train",
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
        finite_loss(float(loss.item()), "train")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def dataset_context(dataset_dir: Path) -> dict:
    readiness = read_json(dataset_dir / "data_readiness.json")
    splits = readiness["splits"]
    return {
        "dataset_source": readiness.get("dataset_source", ""),
        "dataset_revision_or_hash": readiness.get("dataset_revision_or_hash", ""),
        "train_nonempty_rows": splits["train"]["nonempty_rows"],
        "validation_nonempty_rows": splits["validation"]["nonempty_rows"],
        "test_nonempty_rows": splits["test"]["nonempty_rows"],
    }


def run_eval(config: dict, output_dir: Path, device: torch.device) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_runtime(config, output_dir, device)
    tokenizer = ByteTokenizer()
    data_ctx = dataset_context(args.dataset_dir)
    train_blocks = build_blocks(args.dataset_dir, "train", args.sequence_length, tokenizer)
    validation_blocks = build_blocks(args.dataset_dir, "validation", args.sequence_length, tokenizer)
    test_blocks = build_blocks(args.dataset_dir, "test", args.sequence_length, tokenizer)
    certificate = args.graph_certificate
    graph = args.graph_artifact
    records: list[dict] = []
    metrics_rows: list[dict] = []

    for method in args.methods:
        run_dir = output_dir / f"seed{args.seed}_{method}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config_snapshot.json", config)
        write_command(run_dir / "command.sh")
        started = time.perf_counter()
        try:
            torch.manual_seed(args.seed)
            model, artifacts, backend = build_model_and_artifacts(args, method, tokenizer, device)
            train_steps(model, artifacts, train_blocks, args, device)
            validation = run_eval_batches(
                model, artifacts, validation_blocks, args, tokenizer, device, "validation"
            )
            test = run_eval_batches(model, artifacts, test_blocks, args, tokenizer, device, "test")
            finite_loss(float(validation["validation_loss"]), method)
            finite_loss(float(test["test_loss"]), method)
            elapsed = max(time.perf_counter() - started, 1e-9)
            cert_fields = method_certification_fields(method, certificate, args.multiplicity_mode)
            tokens_per_sec = (
                args.eval_batches
                * args.batch_size
                * args.sequence_length
                * 2
                / elapsed
            )
            record = {
                "version": args.version,
                "run_id": f"seed{args.seed}_{method}",
                "status": "ok",
                "failure_reason": "",
                "dataset": "wikitext2",
                "dataset_source": data_ctx["dataset_source"],
                "dataset_revision_or_hash": data_ctx["dataset_revision_or_hash"],
                "tokenizer": tokenizer.name,
                "vocab_size": tokenizer.vocab_size,
                "sequence_length": args.sequence_length,
                "T": args.T,
                "method": method,
                "seed": args.seed,
                "graph_id": args.graph_id,
                "graph_seed": args.graph_seed,
                "B": args.block_size,
                "d": args.degree,
                "q": graph.get("q", ""),
                "G_type": graph.get("G", {}).get("type", ""),
                "H_type": graph.get("H", {}).get("type", ""),
                "causal": args.causal,
                "multiplicity_mode": args.multiplicity_mode,
                "graph_certified": cert_fields["graph_certified"],
                "implementation_certified": cert_fields["implementation_certified"],
                "theory_aligned_method": cert_fields["theory_aligned_method"],
                "architecture": args.architecture,
                "layers": args.layers,
                "d_model": args.d_model,
                "heads": args.heads,
                "ffn_dim": args.ffn_dim,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "eval_batches": args.eval_batches,
                "learning_rate": args.learning_rate,
                "log_every": args.log_every,
                "eval_every": args.eval_every,
                **data_ctx,
                "validation_loss": validation["validation_loss"],
                "validation_perplexity": validation["validation_perplexity"],
                "test_loss": test["test_loss"],
                "test_perplexity": test["test_perplexity"],
                "tokens_per_sec": tokens_per_sec,
                "elapsed_sec": elapsed,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0,
                "pipeline_only": args.steps <= 0,
                "artifact_dir": str(run_dir),
                "git_commit": "",
                "config_sha256": "",
            }
            write_json(run_dir / "summary.json", record)
            records.append(record)
            metrics_rows.append(
                {
                    "run_id": record["run_id"],
                    "method": method,
                    "backend": backend,
                    "validation_loss": record["validation_loss"],
                    "test_loss": record["test_loss"],
                    "effective_K_mean_after_causal": artifacts.metrics.get("effective_k_mean", ""),
                    "attention_pair_count_after_causal": artifacts.metrics.get("attention_pair_count", ""),
                }
            )
        except Exception as exc:
            record = {
                "version": args.version,
                "run_id": f"seed{args.seed}_{method}",
                "status": "failed",
                "failure_reason": repr(exc),
                "dataset": "wikitext2",
                "method": method,
                "seed": args.seed,
                "pipeline_only": args.steps <= 0,
                "artifact_dir": str(run_dir),
            }
            write_json(run_dir / "summary.json", record)
            records.append(record)

    write_csv(output_dir / "results.csv", records, RESULT_FIELDS)
    write_jsonl(output_dir / "results.jsonl", records)
    write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
    readiness_src = args.dataset_dir / "data_readiness.json"
    if readiness_src.exists():
        shutil.copyfile(readiness_src, output_dir / "data_readiness.json")
    summary = {
        "status": "ok" if all(row.get("status") == "ok" for row in records) else "failed",
        "pipeline_only": args.steps <= 0,
        "results": records,
        "metrics": metrics_rows,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_dir = args.output_dir or Path(config["output"]["root"])
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    write_json(output_dir / "config_snapshot.json", config)
    write_command(output_dir / "command.sh")
    summary = run_eval(config, output_dir, device)
    print(json.dumps({"status": summary["status"], "pipeline_only": summary["pipeline_only"]}), flush=True)
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
