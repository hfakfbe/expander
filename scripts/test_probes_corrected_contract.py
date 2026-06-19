from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_tasks import JsonlStore, load_encoder, make_probe_batch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(path: Path, count: int = 4) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= count:
                break
    return rows


def check_task(record: dict) -> dict:
    task = record["task"]
    data_dir = Path(record["version_path"])
    require((data_dir / "train.jsonl").exists(), f"{task}: train.jsonl missing")
    require((data_dir / "test.jsonl").exists(), f"{task}: test.jsonl missing")
    require(not (data_dir / "validation.jsonl").exists(), f"{task}: validation.jsonl must be absent")
    require(bool(record.get("no_target_append_v01")), f"{task}: no_target_append_v01 missing")
    require(bool(record.get("valid_as_test_v01")), f"{task}: valid_as_test_v01 missing")
    require(record["resolved_readout_start"] == "not_applicable_no_external_readout", f"{task}: external readout_start must be disabled")
    require(int(record["resolved_runtime_input_length"]) == int(record["resolved_padded_sequence_length"]), f"{task}: T must equal original input max length")
    card = read_json(data_dir / "dataset_card.json")
    require(card["sha256"]["test_sha256"] == card["sha256"]["source_validation_sha256"], f"{task}: test must equal source validation")
    require(card["sha256"]["source_test_sha256_discarded"] == record["discarded_old_test_sha256"], f"{task}: discarded source test sha mismatch")

    encoder = load_encoder(Path(record["resolved_tokenizer_or_encoder_path"]))
    rows = sample_rows(data_dir / "train.jsonl", 4) + sample_rows(data_dir / "test.jsonl", 4)
    batch = make_probe_batch(rows, record, encoder, torch.device("cpu"))
    require(batch.tokens.shape[1] == int(record["resolved_padded_sequence_length"]), f"{task}: token tensor length mismatch")
    require(batch.tokens.shape[1] == int(record["resolved_runtime_input_length"]), f"{task}: token tensor includes extra slots")
    if task == "lra_listops":
        require(batch.target_positions is None, f"{task}: classification must not create target_positions")
        require(batch.class_targets is not None, f"{task}: classification targets missing")
    else:
        require(batch.target_positions is not None, f"{task}: sequence target_positions missing")
        require(batch.target_mask is not None, f"{task}: target_mask missing")
        valid_positions = batch.target_positions[batch.target_mask]
        require(bool((valid_positions < batch.tokens.shape[1]).all().item()), f"{task}: target position outside input tensor")
        require(bool((valid_positions >= 0).all().item()), f"{task}: negative target position")
        if task == "selective_copy":
            expected_start = int(record["resolved_runtime_input_length"]) - int(record["resolved_runtime_target_length"])
            first_positions = batch.target_positions[0, : int(record["resolved_runtime_target_length"])].tolist()
            require(first_positions == list(range(expected_start, int(record["resolved_runtime_input_length"]))), f"{task}: target positions are not original marker tail")
        if task == "induction_associative_recall":
            raw_positions = [int(item["position"]) for item in rows[0]["target"]]
            first_positions = batch.target_positions[0, : len(raw_positions)].tolist()
            require(first_positions == raw_positions, f"{task}: target positions do not match JSON target.position")
    return {
        "task": task,
        "status": "passed",
        "input_length": int(record["resolved_runtime_input_length"]),
        "target_length": int(record["resolved_runtime_target_length"]),
        "train_rows": int(card["stats"]["generated_train"]["rows"]),
        "test_rows": int(card["stats"]["source_validation_as_test"]["rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/probes_corrected_valid_as_test_l8_log5_task_parameters.json"))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    results = [check_task(record) for record in manifest["tasks"]]
    blocked = manifest.get("blocked_tasks", [])
    for item in blocked:
        require(item.get("status") == "blocked_by_input_contract", f"blocked task malformed: {item.get('task')}")
    out = {
        "status": "passed",
        "tasks": results,
        "blocked_tasks": [{"task": item.get("task"), "reason": item.get("reason")} for item in blocked],
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
