from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OLD_TRAIN_SHA256 = "a5a4aa651a5bdec25075930d1f59b7d0358e29dcca0fdd8f8dc897d55ee3de1c"
OLD_VALIDATION_SHA256 = "e5f48fc67dd3b4c41c39a224b3a01c5f01de7023425fe951875c029c04d82abd"
OLD_TEST_SHA256 = "50de40e9b6f7c53af8a912cf0967ae1129e84028bcc7f90c14a94620d0760fac"
VARIANT = "s4_copying_iid_m1024_marker_readout"
SOURCE_LENGTH = 1024
TARGET_LENGTH = 1024
RAW_SEQUENCE_LENGTH = 2048
MARKER_TOKEN_ID = 63


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_content_payload(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    numeric_meta = {
        key: meta.get(key)
        for key in ["input_length", "target_length", "l_memorize", "l_noise", "n_tokens", "seed"]
        if key in meta
    }
    return {"input": row.get("input"), "target": row.get("target"), "metadata": numeric_meta}


def content_digest(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical_json(normalized_content_payload(row)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def assert_row_contract(row: dict[str, Any], split: str, line_no: int) -> None:
    where = f"{split}:{line_no}"
    tokens = row.get("input")
    target = row.get("target")
    if not isinstance(tokens, list) or not isinstance(target, list):
        raise ValueError(f"{where}: input and target must be lists")
    if len(tokens) != RAW_SEQUENCE_LENGTH:
        raise ValueError(f"{where}: input length {len(tokens)} != {RAW_SEQUENCE_LENGTH}")
    if len(target) != TARGET_LENGTH:
        raise ValueError(f"{where}: target length {len(target)} != {TARGET_LENGTH}")
    if tokens[:SOURCE_LENGTH] != target:
        raise ValueError(f"{where}: input prefix does not equal target")
    if tokens[SOURCE_LENGTH:] != [MARKER_TOKEN_ID] * TARGET_LENGTH:
        raise ValueError(f"{where}: marker suffix is not all {MARKER_TOKEN_ID}")
    if any(not isinstance(v, int) or v < 1 or v > 63 for v in tokens):
        raise ValueError(f"{where}: input token outside [1, 63]")
    if any(not isinstance(v, int) or v < 1 or v > 62 for v in target):
        raise ValueError(f"{where}: target token outside [1, 62]")
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    expected = {
        "input_length": RAW_SEQUENCE_LENGTH,
        "target_length": TARGET_LENGTH,
        "l_memorize": SOURCE_LENGTH,
        "l_noise": 0,
        "n_tokens": 64,
    }
    for key, expected_value in expected.items():
        if int(meta.get(key, -1)) != expected_value:
            raise ValueError(f"{where}: metadata.{key}={meta.get(key)!r} != {expected_value}")


def transform_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, src in enumerate(rows, 1):
        row = json.loads(json.dumps(src))
        if split == "test":
            row["id"] = str(row.get("id", f"copy.validation.{line_no - 1}")).replace(".validation.", ".test.")
        row["variant"] = VARIANT
        meta = row.get("metadata")
        if isinstance(meta, dict):
            meta["variant"] = VARIANT
            meta["split"] = split
        assert_row_contract(row, split, line_no)
        out.append(row)
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(canonical_json(row) + "\n")


def validate_split_level(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    if len(train_rows) != 10_000:
        raise ValueError(f"train row count {len(train_rows)} != 10000")
    if len(test_rows) != 1_000:
        raise ValueError(f"test row count {len(test_rows)} != 1000")
    train_ids = [str(row.get("id")) for row in train_rows]
    test_ids = [str(row.get("id")) for row in test_rows]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("train ids are not unique")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("test ids are not unique")
    if set(train_ids) & set(test_ids):
        raise ValueError("train/test ids overlap")
    train_targets = {tuple(row["target"]) for row in train_rows}
    test_targets = {tuple(row["target"]) for row in test_rows}
    if len(train_targets) != len(train_rows):
        raise ValueError("train target sequences contain duplicates")
    if len(test_targets) != len(test_rows):
        raise ValueError("test target sequences contain duplicates")
    if train_targets & test_targets:
        raise ValueError("train/test target sequences overlap")


def write_checksums(output_dir: Path, controlled_files: list[Path]) -> None:
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_dir)}\n" for path in sorted(controlled_files)),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Users/sxye/Documents/expander_bench/data/probes/copy/s4_copying_length_extrapolation/copy_s4_l0_m1024_a64_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/copy"))
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    source_train = source_dir / "train.jsonl"
    source_validation = source_dir / "validation.jsonl"
    source_test = source_dir / "test.jsonl"
    actual = {
        "train": sha256_file(source_train),
        "validation": sha256_file(source_validation),
        "discarded_test": sha256_file(source_test),
    }
    expected = {
        "train": OLD_TRAIN_SHA256,
        "validation": OLD_VALIDATION_SHA256,
        "discarded_test": OLD_TEST_SHA256,
    }
    if actual != expected:
        raise SystemExit(f"source checksum mismatch: actual={actual} expected={expected}")

    old_train = load_jsonl(source_train)
    old_validation = load_jsonl(source_validation)
    train_rows = transform_rows(old_train, "train")
    test_rows = transform_rows(old_validation, "test")
    validate_split_level(train_rows, test_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    if (output_dir / "validation.jsonl").exists():
        raise SystemExit(f"{output_dir / 'validation.jsonl'} must not exist")

    generated = {
        "train_sha256": sha256_file(output_dir / "train.jsonl"),
        "test_sha256": sha256_file(output_dir / "test.jsonl"),
        "train_content_sha256": content_digest(train_rows),
        "test_content_sha256": content_digest(test_rows),
        "old_train_content_sha256": content_digest(old_train),
        "old_validation_content_sha256": content_digest(old_validation),
    }
    if generated["train_content_sha256"] != generated["old_train_content_sha256"]:
        raise SystemExit("train content digest changed")
    if generated["test_content_sha256"] != generated["old_validation_content_sha256"]:
        raise SystemExit("test content digest changed")

    timestamp = datetime.now(timezone.utc).isoformat()
    write_json(
        output_dir / "dataset_card.json",
        {
            "contract": {
                "encoding": "identity_integer",
                "marker_token_id": MARKER_TOKEN_ID,
                "raw_sequence_length": RAW_SEQUENCE_LENGTH,
                "readout": "marker_positions_1024_2047",
                "source_length": SOURCE_LENGTH,
                "target_length": TARGET_LENGTH,
                "tensor_padding": "none",
                "token_output_size": 64,
                "validation_split": "absent",
                "vocab_size": 64,
            },
            "created_at": timestamp,
            "generated_sha256": generated,
            "rows": {"test": len(test_rows), "train": len(train_rows)},
            "source": {
                "discarded_test_sha256": OLD_TEST_SHA256,
                "source_dir": str(source_dir),
                "train_sha256": OLD_TRAIN_SHA256,
                "validation_as_test_sha256": OLD_VALIDATION_SHA256,
            },
            "task": "copy",
            "variant": VARIANT,
            "version": "copy_corrected_v01",
        },
    )
    write_json(
        output_dir / "source.lock",
        {
            "content_digest_definition": "sha256 canonical JSONL of input,target,numeric metadata only; ignores id and variant",
            "created_at": timestamp,
            "source_dir": str(source_dir),
            "source_files": {
                "discarded_test": {"path": str(source_test), "sha256": OLD_TEST_SHA256, "usage": "blocked"},
                "train": {"path": str(source_train), "rows": len(old_train), "sha256": OLD_TRAIN_SHA256},
                "validation_used_as_test": {
                    "path": str(source_validation),
                    "rows": len(old_validation),
                    "sha256": OLD_VALIDATION_SHA256,
                },
            },
            "transform": {
                "allowed_changes": ["test id .validation. -> .test.", f"variant -> {VARIANT}", "metadata split/variant normalization"],
                "forbidden_changes": ["input array", "target array", "numeric task metadata"],
                "script": "scripts/materialize_copy_corrected.py",
            },
        },
    )
    (output_dir / "config.yaml").write_text(
        "\n".join(
            [
                "version: copy_corrected_v01",
                f"variant: {VARIANT}",
                "source_length: 1024",
                "marker_length: 1024",
                "target_length: 1024",
                "raw_sequence_length: 2048",
                "marker_token_id: 63",
                "vocab_size: 64",
                "token_output_size: 64",
                "encoding: identity_integer",
                "validation_split: absent",
                "test_source: old_validation_jsonl",
                "discarded_old_test: blocked",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Corrected Copy dataset v01\n\n"
        "Canonical local data for copy_corrected_v01. train.jsonl comes from the old train split; "
        "test.jsonl comes from the old validation split; validation.jsonl is intentionally absent. "
        "Inputs are 1024 source tokens plus 1024 marker tokens (63). Labels are read from marker "
        "positions 1024..2047. Token IDs use identity integer encoding with vocab_size=64.\n\n"
        "Raw JSONL files are ignored by Git; recreate them with: python scripts/materialize_copy_corrected.py\n",
        encoding="utf-8",
    )
    write_checksums(
        output_dir,
        [
            output_dir / "README.md",
            output_dir / "dataset_card.json",
            output_dir / "source.lock",
            output_dir / "config.yaml",
            output_dir / "train.jsonl",
            output_dir / "test.jsonl",
        ],
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), **generated}, sort_keys=True))


if __name__ == "__main__":
    main()
