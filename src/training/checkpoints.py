from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, *, model, optimizer, step: int, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": int(step),
            "config": config,
            "torch_rng_state": torch.random.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path, *, model, optimizer=None, map_location="cpu") -> int:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("torch_rng_state") is not None:
        torch.random.set_rng_state(payload["torch_rng_state"])
    return int(payload.get("step", 0))

