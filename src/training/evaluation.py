from __future__ import annotations

import torch

from src.model.backends import BackendBundle
from src.tasks.common import JsonlDataset, TaskSpec
from src.tasks.registry import get_batcher, get_loss


def evaluate(
    *,
    model,
    backends: BackendBundle,
    dataset: JsonlDataset,
    spec: TaskSpec,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
) -> dict:
    model.eval()
    loss_fn = get_loss(spec.name)
    batcher = get_batcher(spec.name)
    totals: dict[str, float] = {"loss_sum": 0.0, "examples": 0.0, "tokens": 0.0}
    metric_sums: dict[str, float] = {}
    batches_seen = 0
    with torch.no_grad():
        for rows in dataset.batches(batch_size, shuffle=False, seed=0):
            batch = batcher(rows, spec, device)
            token_logits, class_logits = model(batch.tokens, batch.pad_mask, backends)
            loss, metrics = loss_fn(token_logits, class_logits, batch, spec)
            examples = float(metrics.get("examples", batch.example_count))
            tokens = float(metrics.get("tokens", batch.token_count))
            totals["loss_sum"] += float(loss.item()) * max(tokens, examples, 1.0)
            totals["examples"] += examples
            totals["tokens"] += tokens
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in {"examples", "tokens", "loss"}:
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * examples
            batches_seen += 1
            if max_batches is not None and batches_seen >= int(max_batches):
                break
    model.train()
    examples = max(totals["examples"], 1.0)
    denom = max(totals["tokens"], examples, 1.0)
    out = {
        "loss": totals["loss_sum"] / denom,
        "examples": int(totals["examples"]),
        "tokens": int(totals["tokens"]),
        "batches": batches_seen,
    }
    for key, value in metric_sums.items():
        out[key] = value / examples
    return out

