import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def write_tiny_listops(repo_dir: Path, rows: int) -> None:
    data_dir = repo_dir / "datasets" / "lra_release" / "listops-1000"
    data_dir.mkdir(parents=True, exist_ok=True)
    examples = [
        ("[ MAX 1 2 3 ]", 3),
        ("[ MIN 4 2 7 ]", 2),
        ("[ SUM_MOD 4 5 6 ]", 5),
        ("[ MEDIAN 1 9 3 ]", 3),
        ("[ MAX [ MIN 8 2 ] 5 ]", 5),
        ("[ SUM_MOD [ MAX 1 4 ] 3 ]", 7),
        ("[ MIN [ SUM_MOD 2 3 ] 6 ]", 5),
        ("[ MEDIAN [ MIN 9 1 ] 4 8 ]", 4),
    ]
    lines = ["Source\tTarget\n"]
    for i in range(rows):
        source, target = examples[i % len(examples)]
        lines.append(f"{source}\t{target}\n")
    for name in ("basic_train.tsv", "basic_val.tsv"):
        (data_dir / name).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    support_dir = Path(__file__).resolve().parent / "smoke_support"
    sys.path.insert(0, str(support_dir))
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    write_tiny_listops(repo_dir, args.rows)

    from lra_config import get_listops_config
    from lra_datasets import ListOpsDataset
    from run_model import get_model, transformers_collator

    config, model_config = get_listops_config()
    config.batch_size = args.batch_size
    config.max_length = args.max_length
    config.total_train_samples = args.steps * args.batch_size
    config.total_eval_samples = args.batch_size * 4
    config.eval_frequency = 0
    config.learning_rate = 1e-3
    config.weight_decay = 0.0

    model_config.max_position_embeddings = args.max_length
    model_config.hidden_size = args.hidden_size
    model_config.num_hidden_layers = args.layers
    model_config.num_attention_heads = args.heads
    model_config.intermediate_size = args.hidden_size * 4

    dataset = ListOpsDataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=transformers_collator,
    )
    model = get_model(config, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses = []
    start = time.perf_counter()
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            inputs, target = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, target = next(it)

        inputs = {key: value.to(device) for key, value in inputs.items()}
        target = target.to(device)
        opt.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step} loss={losses[-1]:.6f}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = args.steps * args.batch_size * args.max_length
    result = {
        "status": "ok",
        "repo_dir": str(repo_dir),
        "steps": args.steps,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "final_loss": losses[-1],
        "mean_loss": float(np.mean(losses)),
        "tokens_per_sec": tokens / elapsed,
        "elapsed_sec": elapsed,
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
