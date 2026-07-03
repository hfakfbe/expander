from __future__ import annotations

from pathlib import Path

from sweep_summary_common import write_random_sweep_summary


ROOT = Path("outputs/copy_corrected_q32_B64_d32_l8_log5/runs")
TRIALS = [
    "q32_B64_d32_l8_log5_random_density50",
    "q32_B64_d32_l8_log5_random_density80",
    "q32_B64_d32_l8_log5_random_density90",
]


def main() -> None:
    out = write_random_sweep_summary(
        root=ROOT,
        trials=TRIALS,
        output_name="random_density_sweep_results.csv",
        missing_fields={
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
        },
    )
    print(out)


if __name__ == "__main__":
    main()
