#!/usr/bin/env python3
"""Plot all v02 probe loss curves by task."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("outputs/probes_dense_to_one_easy_v02/runs")
OUT_DIR = Path("reports/probes_dense_to_one_easy_v02_loss_curves")
TASK_ORDER = [
    "selective_copy",
    "induction_associative_recall",
    "lra_listops",
]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _load_runs() -> dict[str, list[dict[str, object]]]:
    runs_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for results_path in sorted(ROOT.glob("dense_to_one_easy_v02_*/results_train.csv")):
        with results_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                metrics_path = Path(row["metrics_path"])
                if "smoke" in metrics_path.parts:
                    continue
                metrics = _read_jsonl(metrics_path)
                task = row["task"]
                method = row["method"]
                density = _float_or_none(row.get("random_actual_mask_density"))
                if method == "dense":
                    label = "dense"
                    sort_key = (0, 0.0, method)
                else:
                    density_label = f"{density:.8f}".rstrip("0").rstrip(".")
                    method_label = method.replace("random_", "")
                    label = f"{method_label} d={density_label}"
                    method_rank = 1 if method == "random_regular" else 2
                    sort_key = (method_rank, -(density or 0.0), method)
                runs_by_task[task].append(
                    {
                        "density": density,
                        "label": label,
                        "method": method,
                        "metrics": metrics,
                        "metrics_path": metrics_path,
                        "sort_key": sort_key,
                        "steps_completed": int(float(row["steps_completed"])),
                        "test_loss": float(row["test_loss"]),
                    }
                )
    for task_runs in runs_by_task.values():
        task_runs.sort(key=lambda item: item["sort_key"])
    return runs_by_task


def _color(run: dict[str, object]) -> str:
    method = str(run["method"])
    density = run["density"]
    if method == "dense":
        return "#111111"
    if method == "random_regular":
        palette = {
            0.10009765625: "#08306b",
            0.050048828125: "#2171b5",
            0.02490234375: "#6baed6",
            0.015625: "#bdd7e7",
        }
    else:
        palette = {
            0.10009765625: "#67000d",
            0.050048828125: "#cb181d",
            0.02490234375: "#fb6a4a",
            0.015625: "#fcae91",
        }
    return palette.get(density, "#666666")


def _linestyle(run: dict[str, object]) -> str:
    method = str(run["method"])
    if method == "dense":
        return "-"
    if method == "random_regular":
        return "-"
    return "--"


def _plot_metric(ax, runs: list[dict[str, object]], metric_name: str, title: str) -> None:
    for run in runs:
        metrics = run["metrics"]
        steps = [int(item["step"]) for item in metrics if item.get(metric_name) is not None]
        values = [float(item[metric_name]) for item in metrics if item.get(metric_name) is not None]
        ax.plot(
            steps,
            values,
            color=_color(run),
            linestyle=_linestyle(run),
            linewidth=2.0 if run["method"] == "dense" else 1.6,
            label=str(run["label"]),
            marker="o",
            markersize=2.8,
        )
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(metric_name)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def plot_all() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_task = _load_runs()
    written = []
    for task in TASK_ORDER:
        runs = runs_by_task[task]
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{task}: all loss curves", fontsize=14, fontweight="bold")
        _plot_metric(axes[0], runs, "eval_loss", "diagnostic eval loss")
        _plot_metric(axes[1], runs, "train_loss_last_step", "train loss at logged step")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=9,
            ncol=3,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
        output_path = OUT_DIR / f"{task}_all_loss_curves.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        written.append(output_path)
    return written


def main() -> None:
    for path in plot_all():
        print(path)


if __name__ == "__main__":
    main()
