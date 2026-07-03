from __future__ import annotations

from pathlib import Path

from src.io.json import write_json, write_jsonl


def append_metric(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        import json

        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_final_metrics(path: Path, metrics: dict) -> None:
    write_json(path, metrics)


def write_metric_rows(path: Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)

