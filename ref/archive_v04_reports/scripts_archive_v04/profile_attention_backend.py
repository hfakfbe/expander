import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from synthetic_mvp import (
    TaskSpec,
    TinyTransformer,
    build_attention_mask,
    build_cross_mask,
    cross_neighbors_to_block_pair_index,
    make_batch,
    mask_metrics,
    mask_to_neighbors,
    set_seed,
)


def resolve_backend(requested: str, method: str) -> str:
    if requested == "auto":
        return "dense_mask" if method == "dense" else "neighbor"
    if requested == "auto_split":
        return "dense_mask" if method == "dense" else "split"
    if requested == "auto_blockpair":
        return "dense_mask" if method == "dense" else "blockpair"
    if requested in {"neighbor", "split", "blockpair"} and method == "dense":
        raise ValueError(
            "dense method with sparse backend would use K=N; use dense_mask, auto_split, or auto_blockpair"
        )
    return requested


def make_neighbor_tables(method: str, backend: str, args, mask: torch.Tensor, device: torch.device):
    if backend == "neighbor":
        neighbors, valid_neighbors = mask_to_neighbors(mask)
        return neighbors, valid_neighbors, None
    if backend in {"split", "blockpair"}:
        cross_mask = build_cross_mask(method, args.seq_len, args.block_size, args.degree, device, args.seed)
        neighbors, valid_neighbors = mask_to_neighbors(cross_mask)
        block_pair_index = None
        if backend == "blockpair":
            block_pair_index = cross_neighbors_to_block_pair_index(
                neighbors, valid_neighbors, args.block_size
            )
        return neighbors, valid_neighbors, block_pair_index
    return None, None, None


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_method(method: str, args, device: torch.device) -> dict:
    set_seed(args.seed)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    backend = resolve_backend(args.attention_backend, method)
    spec = TaskSpec(num_keys=args.num_keys, num_values=args.num_values)
    mask = build_attention_mask(method, args.seq_len, args.block_size, args.degree, device, args.seed)
    neighbors, valid_neighbors, block_pair_index = make_neighbor_tables(
        method, backend, args, mask, device
    )
    model = TinyTransformer(
        vocab_size=spec.vocab_size,
        num_classes=spec.num_values,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attention_backend=backend,
        block_size=args.block_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def step_once() -> float:
        x, y = make_batch(args, spec, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, mask, neighbors, valid_neighbors, block_pair_index)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{method} produced non-finite loss")
        loss.backward()
        opt.step()
        return float(loss.item())

    for _ in range(args.warmup_steps):
        step_once()
    cuda_sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses = []
    start = time.perf_counter()
    for _ in range(args.measure_steps):
        losses.append(step_once())
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    tokens = args.measure_steps * args.batch_size * args.seq_len

    return {
        "method": method,
        "attention_backend": backend,
        "seq_len": args.seq_len,
        "block_size": args.block_size,
        "degree": args.degree,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens / elapsed,
        "mean_loss": float(statistics.fmean(losses)),
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
        ),
        "mask": mask_metrics(mask, method, args.block_size, args.degree),
        "neighbor_shape": list(neighbors.shape) if neighbors is not None else None,
        "block_pair_shape": list(block_pair_index.shape) if block_pair_index is not None else None,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["attention_backend"]), []).append(row)
    out = []
    for (method, backend), items in grouped.items():
        tokens = [item["tokens_per_sec"] for item in items]
        allocated = [item["peak_allocated_gb"] for item in items]
        reserved = [item["peak_reserved_gb"] for item in items]
        out.append(
            {
                "method": method,
                "attention_backend": backend,
                "repeats": len(items),
                "tokens_per_sec_mean": statistics.fmean(tokens),
                "tokens_per_sec_std": statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
                "peak_allocated_gb_mean": statistics.fmean(allocated),
                "peak_allocated_gb_std": statistics.stdev(allocated) if len(allocated) > 1 else 0.0,
                "peak_reserved_gb_mean": statistics.fmean(reserved),
                "peak_reserved_gb_std": statistics.stdev(reserved) if len(reserved) > 1 else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="copy_first", choices=["copy_first", "copy_visible", "associative_recall"])
    parser.add_argument("--methods", default="dense,local,random,zigzag")
    parser.add_argument(
        "--attention-backend",
        default="auto_split",
        choices=["dense_mask", "neighbor", "split", "blockpair", "auto", "auto_split", "auto_blockpair"],
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-keys", type=int, default=64)
    parser.add_argument("--num-values", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows = []
    for repeat in range(args.repeats):
        for method in methods:
            row = profile_method(method, args, device)
            row["repeat"] = repeat
            rows.append(row)
            print(json.dumps(row), flush=True)

    summary = {
        "status": "ok",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "results": rows,
        "aggregate": aggregate(rows),
    }
    (args.output_dir / "profile.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "profile_runs.csv", rows)
    write_csv(args.output_dir / "profile_summary.csv", summary["aggregate"])


if __name__ == "__main__":
    main()
