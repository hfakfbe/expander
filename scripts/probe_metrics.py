from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def masked_sequence_loss(logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = target_mask.reshape(-1)
    if not bool(flat_mask.any()):
        raise ValueError("sequence target mask is empty")
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


def sequence_metrics(logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = (pred == targets) & target_mask
    token_total = int(target_mask.sum().item())
    token_correct = int(correct.sum().item())
    per_seq_ok = ((pred == targets) | ~target_mask).all(dim=1)
    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "sequence_accuracy": float(per_seq_ok.float().mean().item()) if pred.shape[0] else 0.0,
        "exact_match": float(per_seq_ok.float().mean().item()) if pred.shape[0] else 0.0,
        "token_count": token_total,
    }


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, class_count: int) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = pred == targets
    accuracy = float(correct.float().mean().item()) if targets.numel() else 0.0
    macro_values = []
    for label in range(class_count):
        mask = targets == label
        if bool(mask.any()):
            macro_values.append(float((pred[mask] == targets[mask]).float().mean().item()))
    return {
        "accuracy": accuracy,
        "macro_accuracy": sum(macro_values) / max(len(macro_values), 1),
        "class_count": int(class_count),
    }


def aggregate_metric_rows(rows: list[dict], primary_metric: str) -> dict:
    if not rows:
        return {
            "loss": math.nan,
            "primary_metric_value": 0.0,
            "examples": 0,
            "tokens": 0,
            "secondary_metrics": {},
            "task_metrics": {},
        }
    examples = sum(int(row.get("examples", 0)) for row in rows)
    tokens = sum(int(row.get("tokens", 0)) for row in rows)
    loss_numer = sum(float(row.get("loss", 0.0)) * max(int(row.get("tokens", 0)), int(row.get("examples", 0)), 1) for row in rows)
    loss_denom = sum(max(int(row.get("tokens", 0)), int(row.get("examples", 0)), 1) for row in rows)
    merged: dict[str, float] = {}
    weights: dict[str, int] = {}
    for row in rows:
        weight = max(int(row.get("tokens", 0)), int(row.get("examples", 0)), 1)
        for key, value in row.items():
            if key in {"loss", "examples", "tokens", "subtask"}:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                merged[key] = merged.get(key, 0.0) + float(value) * weight
                weights[key] = weights.get(key, 0) + weight
    averaged = {key: merged[key] / max(weights[key], 1) for key in merged}
    subtask_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        subtask = str(row.get("subtask") or "not_applicable")
        subtask_rows[subtask].append(row)
    subtask_metrics = {}
    for subtask, items in subtask_rows.items():
        if subtask == "not_applicable":
            continue
        subtask_metrics[subtask] = aggregate_metric_rows(
            [{key: value for key, value in item.items() if key != "subtask"} for item in items],
            primary_metric,
        )["secondary_metrics"]
    primary = averaged.get(primary_metric, averaged.get("accuracy", averaged.get("exact_match", averaged.get("token_accuracy", 0.0))))
    return {
        "loss": loss_numer / max(loss_denom, 1),
        "primary_metric_value": primary,
        "examples": examples,
        "tokens": tokens,
        "secondary_metrics": averaged,
        "task_metrics": {"subtasks": subtask_metrics},
    }


def write_training_curves(metrics_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in metrics_rows if row.get("split") == "train"]
    panels = [
        ("train_loss", "train loss"),
        ("eval_loss", "eval loss"),
        ("primary_metric_value", "primary metric"),
        ("learning_rate", "learning rate"),
        ("seconds_since_prev_log", "seconds/log"),
        ("examples_per_sec", "examples/sec"),
        ("tokens_per_sec", "tokens/sec"),
        ("peak_allocated_gb", "peak allocated GB"),
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _fallback_training_curves_png(train_rows, path, panels)
        return

    rows = max(1, math.ceil(len(panels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.0, 3.0 * rows), constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    steps = [int(row.get("step", index + 1)) for index, row in enumerate(train_rows)]
    for ax, (key, title) in zip(axes_flat, panels):
        values = []
        for row in train_rows:
            try:
                values.append(float(row.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                values.append(0.0)
        ax.plot(steps, values, marker="o" if len(steps) <= 50 else None, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_png_rgb(path: Path, width: int, height: int, pixels: bytearray) -> None:
    import struct
    import zlib

    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )


def _fallback_training_curves_png(metrics_rows: list[dict], path: Path, panels: list[tuple[str, str]]) -> None:
    cols = 2
    panel_w = 540
    panel_h = 240
    rows = max(1, math.ceil(len(panels) / cols))
    width = cols * panel_w
    height = rows * panel_h
    pixels = bytearray([255] * width * height * 3)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            put(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    steps = [int(row.get("step", idx + 1) or idx + 1) for idx, row in enumerate(metrics_rows)]
    for idx, (key, _title) in enumerate(panels):
        ox = (idx % cols) * panel_w
        oy = (idx // cols) * panel_h
        left, top = ox + 48, oy + 28
        right, bottom = ox + panel_w - 24, oy + panel_h - 32
        line(left, bottom, right, bottom, (40, 40, 40))
        line(left, top, left, bottom, (40, 40, 40))
        line(right, top, right, bottom, (220, 220, 220))
        line(left, top, right, top, (220, 220, 220))
        values = []
        for row in metrics_rows:
            try:
                value = float(row.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            values.append(value if math.isfinite(value) else 0.0)
        if not values:
            continue
        vmin = min(values)
        vmax = max(values)
        if vmax <= vmin:
            vmax = vmin + 1.0
        smin = min(steps)
        smax = max(steps)
        if smax <= smin:
            smax = smin + 1
        points = []
        for step, value in zip(steps, values):
            x = left + round((step - smin) * (right - left) / (smax - smin))
            y = bottom - round((value - vmin) * (bottom - top) / (vmax - vmin))
            points.append((x, y))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            line(x0, y0, x1, y1, (31, 119, 180))
        for x, y in points:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    put(x + dx, y + dy, (214, 39, 40))
    _write_png_rgb(path, width, height, pixels)


def json_metric(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
