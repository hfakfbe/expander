from __future__ import annotations

import csv
import json
from pathlib import Path

from .artifacts import RESULT_FIELDS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else RESULT_FIELDS
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def plot_training_curves(metrics_rows: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in metrics_rows]
    panels = [
        ("train_loss", "train loss"),
        ("eval_loss", "eval loss"),
        ("eval_token_accuracy", "eval token accuracy"),
        ("eval_sequence_accuracy", "eval sequence accuracy"),
        ("eval_eos_accuracy", "eval EOS accuracy"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for ax, (key, title) in zip(axes.flatten(), panels):
        ax.plot(steps, [float(row[key]) for row in metrics_rows], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    for ax in axes.flatten()[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
