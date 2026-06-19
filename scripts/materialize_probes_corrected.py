from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "probes_corrected_valid_as_test_l8_log5"
SOURCE_ROOT = Path("/Users/sxye/Documents/expander_bench/data/probes")

TRAINABLE_TASKS: dict[str, str] = {
    "selective_copy": "selective_copy/s4_variable_copy_regenerated/selective_copy_s4_l4096_m16_a16_full_v1",
    "induction_associative_recall": "induction_associative_recall/zoology_mqar_regenerated/mqar_vocab8192_len64_128_256_512_1024_full_v2",
    "lra_listops": "lra_listops/lra_official_generator_regenerated/lra_listops_regenerated_len500_2000_96k_2k_2k_full_v1",
}

BLOCKED_TASKS: dict[str, dict[str, str]] = {
    "niah_kv_retrieval": {
        "source_relpath": "niah_kv_retrieval/ruler_niah_single_1/niah_ruler_noise_4k_full_v2",
        "reason": "JSONL input ends in a natural-language question; the answer is only in target and no original-input answer/readout slot exists.",
    },
    "ruler": {
        "source_relpath": "ruler/ruler_official_nonqa_synthetic_suite/ruler_nonqa_suite_4k_full_v2",
        "reason": "JSONL input ends in a natural-language question; the answer is only in target and no original-input answer/readout slot exists.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def split_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    input_lengths = []
    target_lengths = []
    target_positions_bad = 0
    target_positions_total = 0
    selective_tail_marker_rows = 0
    for row in rows:
        input_value = row.get("input")
        target_value = row.get("target")
        input_len = len(input_value) if hasattr(input_value, "__len__") else 0
        if isinstance(target_value, (list, str)):
            target_len = len(target_value)
        elif target_value is None:
            target_len = 0
        else:
            target_len = 1
        input_lengths.append(input_len)
        target_lengths.append(target_len)
        if isinstance(target_value, list) and target_value and all(isinstance(item, dict) for item in target_value):
            for item in target_value:
                target_positions_total += 1
                pos = int(item["position"])
                if pos < 0 or pos >= input_len:
                    target_positions_bad += 1
        if (
            isinstance(input_value, list)
            and isinstance(target_value, list)
            and target_value
            and all(not isinstance(item, dict) for item in target_value)
            and input_value[-len(target_value) :] == [15] * len(target_value)
        ):
            selective_tail_marker_rows += 1
    return {
        "rows": len(rows),
        "input_length_min": min(input_lengths) if input_lengths else 0,
        "input_length_max": max(input_lengths) if input_lengths else 0,
        "target_length_min": min(target_lengths) if target_lengths else 0,
        "target_length_max": max(target_lengths) if target_lengths else 0,
        "target_positions_bad": target_positions_bad,
        "target_positions_total": target_positions_total,
        "selective_tail_marker_rows": selective_tail_marker_rows,
    }


def materialize_task(task: str, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_train = source_dir / "train.jsonl"
    source_validation = source_dir / "validation.jsonl"
    source_test = source_dir / "test.jsonl"
    if not source_train.exists() or not source_validation.exists() or not source_test.exists():
        raise FileNotFoundError(f"missing required source splits under {source_dir}")

    train_filter_policy = "byte-identical copy of source train.jsonl"
    validation_rows = read_jsonl(source_validation)
    validation_input_lengths = sorted({len(row.get("input")) for row in validation_rows})
    if task == "induction_associative_recall" and len(validation_input_lengths) == 1:
        required_input_length = int(validation_input_lengths[0])
        source_train_rows = read_jsonl(source_train)
        filtered_rows = [row for row in source_train_rows if len(row.get("input")) == required_input_length]
        if not filtered_rows:
            raise RuntimeError(f"{task}: no train rows match validation input length {required_input_length}")
        write_jsonl(output_dir / "train.jsonl", filtered_rows)
        train_filter_policy = (
            f"filtered source train.jsonl to rows with input length {required_input_length}, "
            "matching source validation used as test"
        )
    else:
        shutil.copyfile(source_train, output_dir / "train.jsonl")
    shutil.copyfile(source_validation, output_dir / "test.jsonl")
    forbidden_validation = output_dir / "validation.jsonl"
    if forbidden_validation.exists():
        forbidden_validation.unlink()

    shas = {
        "source_train_sha256": sha256_file(source_train),
        "source_validation_sha256": sha256_file(source_validation),
        "source_test_sha256_discarded": sha256_file(source_test),
        "train_sha256": sha256_file(output_dir / "train.jsonl"),
        "test_sha256": sha256_file(output_dir / "test.jsonl"),
    }
    if train_filter_policy.startswith("byte-identical") and shas["train_sha256"] != shas["source_train_sha256"]:
        raise RuntimeError(f"{task}: generated train differs from source train")
    if shas["test_sha256"] != shas["source_validation_sha256"]:
        raise RuntimeError(f"{task}: generated test differs from source validation")

    stats = {
        "source_train": split_stats(source_train),
        "generated_train": split_stats(output_dir / "train.jsonl"),
        "source_validation_as_test": split_stats(source_validation),
        "source_test_discarded": split_stats(source_test),
    }
    card = {
        "version": VERSION,
        "task": task,
        "created_at": utc_now(),
        "source_dir": str(source_dir),
        "split_policy": {
            "train": train_filter_policy,
            "test": "byte-identical copy of source validation.jsonl",
            "validation": "absent",
            "discarded_source_test": "source test.jsonl is intentionally not used because train/test must share the same input contract",
        },
        "input_contract": {
            "no_target_append": True,
            "runtime_rule": "model input length is the encoded/original input length only; labels are supervised at original-input positions or by a classifier head",
            "target_storage": "target remains a JSONL label field and is never concatenated to input",
        },
        "sha256": shas,
        "stats": stats,
    }
    write_json(output_dir / "dataset_card.json", card)
    write_json(
        output_dir / "source.lock",
        {
            "version": VERSION,
            "task": task,
            "source_dir": str(source_dir),
            "source_train": str(source_train),
            "source_validation_used_as_test": str(source_validation),
            "source_test_discarded": str(source_test),
            "sha256": shas,
            "created_at": utc_now(),
        },
    )
    write_text(
        output_dir / "checksums.sha256",
        "\n".join(
            [
                f"{sha256_file(output_dir / 'train.jsonl')}  train.jsonl",
                f"{sha256_file(output_dir / 'test.jsonl')}  test.jsonl",
                f"{sha256_file(output_dir / 'dataset_card.json')}  dataset_card.json",
                f"{sha256_file(output_dir / 'source.lock')}  source.lock",
            ]
        )
        + "\n",
    )
    write_text(
        output_dir / "README.md",
        f"# {task} corrected valid-as-test data\n\n"
        f"Version: {VERSION}\n\n"
        "This directory intentionally contains only train.jsonl and test.jsonl.\n"
        "test.jsonl is byte-identical to the source validation.jsonl; the source test.jsonl is discarded.\n"
        "The runtime contract forbids concatenating target labels to the input.\n",
    )
    return {"task": task, "output_dir": str(output_dir), "sha256": shas, "stats": stats}


def blocked_task_audit(task: str, source_dir: Path, reason: str) -> dict[str, Any]:
    sample = read_jsonl(source_dir / "train.jsonl")[0]
    input_value = sample.get("input", "")
    return {
        "task": task,
        "status": "blocked_by_input_contract",
        "source_dir": str(source_dir),
        "reason": reason,
        "source_train_sha256": sha256_file(source_dir / "train.jsonl"),
        "source_validation_sha256": sha256_file(source_dir / "validation.jsonl"),
        "source_test_sha256_discarded": sha256_file(source_dir / "test.jsonl"),
        "sample_target": sample.get("target"),
        "sample_metadata": sample.get("metadata"),
        "sample_input_tail": input_value[-500:] if isinstance(input_value, str) else input_value[-32:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("datasets/probes_corrected_valid_as_test_l8_log5"))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    outputs = [
        materialize_task(task, args.source_root / relpath, args.output_root / task)
        for task, relpath in TRAINABLE_TASKS.items()
    ]
    blocked = [
        blocked_task_audit(task, args.source_root / payload["source_relpath"], payload["reason"])
        for task, payload in BLOCKED_TASKS.items()
    ]
    manifest = {
        "version": VERSION,
        "created_at": utc_now(),
        "output_root": str(args.output_root),
        "trainable_tasks": outputs,
        "blocked_tasks": blocked,
    }
    write_json(args.output_root / "materialization_manifest.json", manifest)
    print(json.dumps({"status": "ok", "manifest": str(args.output_root / "materialization_manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
