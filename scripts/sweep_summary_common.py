from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def scalar(data: dict[str, Any], key: str, default: Any = "") -> Any:
    value = data.get(key, default)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_random_sweep_summary(
    *,
    root: Path,
    trials: list[str],
    output_name: str,
    missing_fields: dict[str, Any],
    extra_fields: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        path = root / trial / "random_regular" / "seed0" / "final_eval.json"
        if not path.exists():
            rows.append(
                {
                    "trial_id": trial,
                    "status": "missing",
                    "method": "random_regular",
                    "seed": 0,
                    **missing_fields,
                    "final_eval_path": str(path),
                }
            )
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("attention_metrics", {})
        rows.append(
            {
                "trial_id": trial,
                "status": scalar(data, "status"),
                "method": scalar(data, "method"),
                "seed": scalar(data, "seed"),
                "requested_density": data.get("identity", {}).get("random_actual_mask_density", ""),
                **(extra_fields(data, metrics) if extra_fields else {}),
                "actual_mask_density": scalar(data, "actual_mask_density"),
                "attention_pair_count": metrics.get("attention_pair_count", ""),
                "unique_k_min": metrics.get("unique_k_min", ""),
                "unique_k_mean": metrics.get("unique_k_mean", ""),
                "unique_k_max": metrics.get("unique_k_max", ""),
                "test_loss": scalar(data, "test_loss"),
                "copy_token_accuracy": scalar(data, "copy_token_accuracy"),
                "copy_sequence_accuracy": scalar(data, "copy_sequence_accuracy"),
                "checkpoint_path": scalar(data, "checkpoint_path"),
                "final_eval_path": str(path),
            }
        )
    out = root / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
