from __future__ import annotations

import math


def learning_rate(config: dict, step: int) -> float:
    training = config["training"]
    scheduler = str(training["scheduler"])
    base = float(training["learning_rate"])
    warmup = int(training["warmup_steps"])
    min_lr = float(training["min_learning_rate"])
    if scheduler == "constant":
        if warmup > 0 and step <= warmup:
            return base * step / max(warmup, 1)
        return base
    if scheduler != "cosine":
        raise ValueError(f"unknown scheduler: {scheduler}")
    total = max(int(training["max_steps"]), 1)
    if warmup > 0 and step <= warmup:
        return base * step / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def apply_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)

