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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=["listops", "imdb"], default="listops")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
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

    from lra_config import get_listops_config, get_text_classification_config
    from lra_datasets import ImdbDataset, ListOpsDataset
    from run_model import get_model, transformers_collator

    if args.task == "listops":
        config, model_config = get_listops_config()
        dataset_cls = ListOpsDataset
    else:
        config, model_config = get_text_classification_config()
        dataset_cls = ImdbDataset
    config.batch_size = args.batch_size
    config.max_length = args.max_length
    config.total_train_samples = args.steps * args.batch_size
    config.total_eval_samples = args.eval_batches * args.batch_size
    config.eval_frequency = 0
    config.learning_rate = 1e-3
    config.weight_decay = 0.0

    model_config.max_position_embeddings = args.max_length
    model_config.hidden_size = args.hidden_size
    model_config.num_hidden_layers = args.layers
    model_config.num_attention_heads = args.heads
    model_config.intermediate_size = args.hidden_size * 4

    train_ds = dataset_cls(config, split="train")
    eval_ds = dataset_cls(config, split="eval")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError(f"{args.task} dataset is empty: train={len(train_ds)} eval={len(eval_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=transformers_collator
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=transformers_collator
    )

    model = get_model(config, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def run_eval() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, target) in enumerate(eval_loader):
                if i >= args.eval_batches:
                    break
                inputs = {key: value.to(device) for key, value in inputs.items()}
                target = target.to(device)
                outputs = model(**inputs)
                loss = F.cross_entropy(outputs.logits, target)
                total_loss += float(loss.item())
                total_correct += int((outputs.logits.argmax(dim=-1) == target).sum().item())
                total += int(target.numel())
        model.train()
        denom = max(1, min(args.eval_batches, len(eval_loader)))
        return total_loss / denom, total_correct / max(1, total)

    losses = []
    metrics_fp = None
    if args.metrics_jsonl is not None:
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        metrics_fp = args.metrics_jsonl.open("w", encoding="utf-8")
    start = time.perf_counter()
    it = iter(train_loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            inputs, target = next(it)
        except StopIteration:
            it = iter(train_loader)
            inputs, target = next(it)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        target = target.to(device)
        opt.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            eval_loss, eval_accuracy = run_eval()
            row = {
                "step": step,
                "train_loss": losses[-1],
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
            }
            print(json.dumps(row), flush=True)
            if metrics_fp is not None:
                metrics_fp.write(json.dumps(row) + "\n")
                metrics_fp.flush()

    eval_loss, eval_accuracy = run_eval()
    if metrics_fp is not None:
        metrics_fp.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = args.steps * args.batch_size * args.max_length
    result = {
        "status": "ok",
        "task": args.task,
        "repo_dir": str(repo_dir),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "heads": args.heads,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "final_train_loss": losses[-1],
        "mean_train_loss": float(np.mean(losses)),
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
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
