from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/copy_corrected_q32_B64_d32_l8_log5/runs")
TRIALS = [
    "q32_B64_d32_l8_log5_random_density50",
    "q32_B64_d32_l8_log5_random_density80",
    "q32_B64_d32_l8_log5_random_density90",
]


def scalar(data: dict[str, Any], key: str, default: Any = "") -> Any:
    value = data.get(key, default)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def main() -> None:
    rows: list[dict[str, Any]] = []
    for trial in TRIALS:
        path = ROOT / trial / "random_regular" / "seed0" / "final_eval.json"
        if not path.exists():
            rows.append(
                {
                    "trial_id": trial,
                    "status": "missing",
                    "method": "random_regular",
                    "seed": 0,
                    "requested_density": "",
                    "actual_mask_density": "",
                    "attention_pair_count": "",
                    "unique_k_min": "",
                    "unique_k_mean": "",
                    "unique_k_max": "",
                    "test_loss": "",
                    "copy_token_accuracy": "",
                    "copy_sequence_accuracy": "",
                    "checkpoint_path": "",
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
    out = ROOT / "random_density_sweep_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
