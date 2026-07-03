from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections.abc import Iterable

from png_utils import draw_line, write_png_rgb

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
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

def _curve_panels(metrics_rows: list[dict]) -> list[tuple[str, str]]:
    available = {key for row in metrics_rows for key in row}
    copy_panels = [
        ("train_loss", "train loss"),
        ("eval_loss", "eval loss"),
        ("learning_rate", "learning rate"),
        ("eval_token_accuracy", "eval token accuracy"),
        ("eval_sequence_accuracy", "eval sequence accuracy"),
        ("eval_eos_accuracy", "eval EOS accuracy"),
        ("seconds_since_prev_log", "seconds since prev log"),
        ("tokens_per_sec", "tokens/sec"),
    ]
    wikitext_panels = [
        ("train_loss", "train loss"),
        ("running_train_perplexity", "running train ppl"),
        ("learning_rate", "learning rate"),
        ("seconds_since_prev_log", "seconds since prev log"),
        ("tokens_per_sec", "tokens/sec"),
        ("test_loss", "test loss"),
        ("test_perplexity", "test perplexity"),
    ]
    if {"eval_token_accuracy", "eval_sequence_accuracy", "eval_eos_accuracy"} & available:
        return copy_panels
    return [panel for panel in wikitext_panels if panel[0] in available]


_FONT_3X5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "a": ("010", "101", "111", "101", "101"),
    "b": ("110", "101", "110", "101", "110"),
    "c": ("011", "100", "100", "100", "011"),
    "d": ("110", "101", "101", "101", "110"),
    "e": ("111", "100", "110", "100", "111"),
    "f": ("111", "100", "110", "100", "100"),
    "g": ("011", "100", "101", "101", "011"),
    "h": ("101", "101", "111", "101", "101"),
    "i": ("111", "010", "010", "010", "111"),
    "j": ("001", "001", "001", "101", "010"),
    "k": ("101", "101", "110", "101", "101"),
    "l": ("100", "100", "100", "100", "111"),
    "m": ("101", "111", "111", "101", "101"),
    "n": ("110", "101", "101", "101", "101"),
    "o": ("010", "101", "101", "101", "010"),
    "p": ("110", "101", "110", "100", "100"),
    "q": ("010", "101", "101", "011", "001"),
    "r": ("110", "101", "110", "101", "101"),
    "s": ("011", "100", "010", "001", "110"),
    "t": ("111", "010", "010", "010", "010"),
    "u": ("101", "101", "101", "101", "111"),
    "v": ("101", "101", "101", "101", "010"),
    "w": ("101", "101", "111", "111", "101"),
    "x": ("101", "101", "010", "101", "101"),
    "y": ("101", "101", "010", "010", "010"),
    "z": ("111", "001", "010", "100", "111"),
    "/": ("001", "001", "010", "100", "100"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    ":": ("000", "010", "000", "010", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


def _fallback_training_curves_png(
    metrics_rows: list[dict],
    output_path: Path,
    panels: list[tuple[str, str]],
) -> None:
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

    def text(x: int, y: int, value: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for char in value.lower():
            glyph = _FONT_3X5.get(char, _FONT_3X5[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        for sy in range(scale):
                            for sx in range(scale):
                                put(cursor + gx * scale + sx, y + gy * scale + sy, color)
            cursor += 4 * scale

    steps = [int(row.get("step", idx + 1) or idx + 1) for idx, row in enumerate(metrics_rows)]
    for idx, (key, title) in enumerate(panels):
        ox = (idx % cols) * panel_w
        oy = (idx // cols) * panel_h
        left, top = ox + 48, oy + 36
        right, bottom = ox + panel_w - 24, oy + panel_h - 32
        text(ox + 16, oy + 12, title[:28], (30, 30, 30))
        draw_line(put, left, bottom, right, bottom, (40, 40, 40))
        draw_line(put, left, top, left, bottom, (40, 40, 40))
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
            draw_line(put, x0, y0, x1, y1, (31, 119, 180))
        for x, y in points:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    put(x + dx, y + dy, (214, 39, 40))
        text(left, bottom + 8, f"{smin}-{smax}", (70, 70, 70), scale=1)
        text(left, top - 12, f"{vmax:.3g}", (70, 70, 70), scale=1)
        text(left, bottom - 8, f"{vmin:.3g}", (70, 70, 70), scale=1)
    write_png_rgb(output_path, width, height, pixels)


def plot_training_curves(metrics_rows: list[dict], output_path: Path) -> None:
    panels = _curve_panels(metrics_rows)
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render training_curves.png. "
            "Install matplotlib in the active experiment environment before plotting."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in metrics_rows]
    rows = max(1, math.ceil(len(panels) / 2))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(12.5, max(3.2, 3.0 * rows)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    if not isinstance(axes, Iterable):
        axes = [axes]
    else:
        axes = [ax for row in axes for ax in (row if isinstance(row, Iterable) else [row])]
    color = "#1f77b4"
    marker = "o" if len(steps) <= 60 else None
    for ax, (key, title) in zip(axes, panels):
        values = []
        for row in metrics_rows:
            try:
                value = float(row.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            values.append(value if math.isfinite(value) else 0.0)
        ax.plot(
            steps,
            values,
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=2.0,
            alpha=0.95,
        )
        ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
        ax.set_xlabel("step")
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b8c0cc")
        ax.spines["bottom"].set_color("#b8c0cc")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)
