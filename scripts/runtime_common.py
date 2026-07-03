from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def state_dict_sha256(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def tensor_checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_coverage(indices: list[int], n: int) -> dict[str, int]:
    counts = Counter(indices)
    return {
        "draw_count": len(indices),
        "unique_count": len(counts),
        "never_seen": int(n) - len(counts),
        "max_repeat_count": max(counts.values()) if counts else 0,
    }


def write_checkpoint_manifest(run_dir: Path, checkpoints: list[dict[str, Any]], policy: str) -> None:
    payload = {
        "checkpoint_policy": policy,
        "tensor_checkpoints_written": checkpoints,
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "checkpoint_files_git_ignored": True,
    }
    path = run_dir / "checkpoint_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
