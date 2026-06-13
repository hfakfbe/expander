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
    read_json,
    write_command,
    write_json,
    write_jsonl,
)


def run_smoke(config: dict, output_dir: Path, device: torch.device) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_runtime(config, output_dir, device)
    tokenizer = ByteTokenizer()
    train_blocks = build_blocks(args.dataset_dir, "train", args.sequence_length, tokenizer)
    val_blocks = build_blocks(args.dataset_dir, "validation", args.sequence_length, tokenizer)
    max_train_batches = int(config.get("smoke", {}).get("max_train_batches", 1))
    max_eval_batches = int(config.get("smoke", {}).get("max_eval_batches", 1))
    backward_method = str(config.get("smoke", {}).get("backward_method", args.methods[0]))

    metrics_rows: list[dict] = []
    examples: list[dict] = []
    for method in args.methods:
        torch.manual_seed(args.seed)
        model, artifacts, backend = build_model_and_artifacts(args, method, tokenizer, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        started = time.perf_counter()
        train_loss = None
        for batch_index in range(max_train_batches):
            batch = make_lm_batch(
                train_blocks,
                args.batch_size,
                batch_index,
                device,
                args.sequence_length,
                args.T,
                args.seed,
                f"smoke_train_{method}",
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
            loss, train_metrics = lm_loss_and_metrics(logits, batch)
            if not math.isfinite(float(loss.item())):
                raise ValueError(f"non-finite smoke loss for {method}")
            train_loss = float(loss.item())
            if method == backward_method and batch_index == 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        eval_loss = 0.0
        eval_tokens = 0
        with torch.no_grad():
            for batch_index in range(max_eval_batches):
                batch = make_lm_batch(
                    val_blocks,
                    args.batch_size,
                    batch_index,
                    device,
                    args.sequence_length,
                    args.T,
                    args.seed,
                    f"smoke_validation_{method}",
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
                eval_loss += float(loss.item()) * batch.targets.numel()
                eval_tokens += batch.targets.numel()
        elapsed = max(time.perf_counter() - started, 1e-9)
        metrics_rows.append(
            {
                "version": args.version,
                "method": method,
                "status": "ok",
                "backend": backend,
                "sequence_length": args.sequence_length,
                "T": args.T,
                "batch_size": args.batch_size,
                "train_loss": train_loss,
                "validation_loss": eval_loss / max(eval_tokens, 1),
                "loss_is_finite": True,
                "backward_ran": method == backward_method,
                "tokens_per_sec": (max_train_batches * args.batch_size * args.sequence_length) / elapsed,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0,
                "effective_K_mean_after_causal": artifacts.metrics.get("effective_k_mean", ""),
                "attention_pair_count_after_causal": artifacts.metrics.get("attention_pair_count", ""),
            }
        )
        if not examples:
            sample = make_lm_batch(
                train_blocks,
                1,
                0,
                torch.device("cpu"),
                args.sequence_length,
                args.T,
                args.seed,
                "example",
            )
            examples.append(
                {
                    "tokens_prefix": sample.tokens[0, :32].tolist(),
                    "targets_prefix": sample.targets[0, :32].tolist(),
                    "sequence_length": args.sequence_length,
                    "T": args.T,
                }
            )

    write_jsonl(output_dir / "smoke_metrics.jsonl", metrics_rows)
    write_jsonl(output_dir / "batch_examples.jsonl", examples)
    readiness_src = args.dataset_dir / "data_readiness.json"
    if readiness_src.exists():
        shutil.copyfile(readiness_src, output_dir / "data_readiness.json")
    summary = {
        "status": "ok",
        "dataset_dir": str(args.dataset_dir),
        "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
        "methods": args.methods,
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
    summary = run_smoke(config, output_dir, device)
    print(json.dumps({"status": summary["status"], "methods": summary["methods"]}), flush=True)


if __name__ == "__main__":
    main()
