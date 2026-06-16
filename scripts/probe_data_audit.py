from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from probe_common import (
    ATTENTION_CONTRACT,
    EXPERIMENT_VERSION,
    REQUIRED_VERSION_FILES,
    SELECTED_PROBES,
    command_string,
    count_lines,
    ensure_no_forbidden_probe,
    file_sha256,
    read_json,
    read_yaml,
    selected_probe_path,
    stats,
    utc_now,
    write_command,
    write_csv,
    write_json,
    write_jsonl,
)


SPLITS = ["train", "validation", "test"]


def _value_type(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "array[empty]"
        item_types = sorted({_value_type(item) for item in value[:32]})
        return "array[" + "|".join(item_types) + "]"
    if isinstance(value, dict):
        return "object{" + ",".join(sorted(value.keys())) + "}"
    return type(value).__name__


def _target_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 1


def _input_length(row: dict) -> int:
    value = row.get("input")
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        meta = row.get("metadata", {})
        if isinstance(meta, dict) and "input_length" in meta:
            return int(meta["input_length"])
        return len(value.encode("utf-8"))
    return 1


def _tokens_from_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if isinstance(item, dict):
                out.extend(item.values())
            else:
                out.append(item)
        return out
    return [value]


def checksum_entries(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        entries.append((expected.strip(), name.strip()))
    return entries


def verify_checksums(version_dir: Path) -> dict:
    rows = []
    all_ok = True
    checksums_path = version_dir / "checksums.sha256"
    for expected, name in checksum_entries(checksums_path):
        file_path = version_dir / name
        exists = file_path.exists()
        actual = file_sha256(file_path) if exists else ""
        ok = exists and actual == expected
        all_ok = all_ok and ok
        rows.append(
            {
                "file": name,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": exists,
                "status": "ok" if ok else "failed",
            }
        )
    return {"status": "ok" if all_ok else "failed", "files": rows}


def scan_split(path: Path, sample_preview: list[dict]) -> dict:
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    metadata_keys: Counter[str] = Counter()
    input_types: Counter[str] = Counter()
    target_types: Counter[str] = Counter()
    token_values: Counter[str] = Counter()
    target_values: Counter[str] = Counter()
    ruler_subtasks: Counter[str] = Counter()
    first_row: dict | None = None
    with path.open("r", encoding="utf-8") as fp:
        for index, line in enumerate(fp):
            if not line.strip():
                continue
            row = json.loads(line)
            if first_row is None:
                first_row = row
                sample_preview.append(
                    {
                        "split_path": str(path),
                        "id": row.get("id"),
                        "task": row.get("task"),
                        "variant": row.get("variant"),
                        "input_type": _value_type(row.get("input")),
                        "input_length": _input_length(row),
                        "input_preview": str(row.get("input"))[:240],
                        "target_type": _value_type(row.get("target")),
                        "target_length": _target_length(row.get("target")),
                        "target_preview": str(row.get("target"))[:240],
                        "metadata": row.get("metadata", {}),
                    }
                )
            input_lengths.append(_input_length(row))
            target_lengths.append(_target_length(row.get("target")))
            input_types[_value_type(row.get("input"))] += 1
            target_types[_value_type(row.get("target"))] += 1
            meta = row.get("metadata", {})
            if isinstance(meta, dict):
                metadata_keys.update(str(key) for key in meta.keys())
                if meta.get("ruler_task"):
                    ruler_subtasks[str(meta["ruler_task"])] += 1
            for token in _tokens_from_value(row.get("input"))[:256]:
                token_values[str(token)] += 1
            for token in _tokens_from_value(row.get("target"))[:256]:
                target_values[str(token)] += 1
    return {
        "rows": len(input_lengths),
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "input_type_counts": dict(input_types),
        "target_type_counts": dict(target_types),
        "metadata_keys": sorted(metadata_keys),
        "token_value_estimate": len(token_values),
        "target_value_estimate": len(target_values),
        "ruler_subtasks": dict(ruler_subtasks),
        "first_row_task": first_row.get("task") if first_row else "",
        "first_row_variant": first_row.get("variant") if first_row else "",
    }


def audit_task(task: str, output_dir: Path, sample_preview: list[dict]) -> tuple[dict, dict]:
    ensure_no_forbidden_probe(task)
    version_dir = selected_probe_path(task)
    missing_files = [name for name in REQUIRED_VERSION_FILES if not (version_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"{task} missing required files: {missing_files}")
    dataset_card = read_json(version_dir / "dataset_card.json")
    deployment_status = read_yaml(version_dir / "deployment_status.yaml")
    checksum = verify_checksums(version_dir)
    split_scans = {
        split: scan_split(version_dir / f"{split}.jsonl", sample_preview)
        for split in SPLITS
    }
    all_input_lengths = [value for split in SPLITS for value in split_scans[split]["input_lengths"]]
    all_target_lengths = [value for split in SPLITS for value in split_scans[split]["target_lengths"]]
    input_stats = stats(all_input_lengths)
    target_stats = stats(all_target_lengths)
    card_contract = dataset_card.get("non_causal_contract", {})
    status_ok = (
        dataset_card.get("status") == "validated"
        and bool(dataset_card.get("can_enter_main_eval")) is True
        and deployment_status.get("status") == "validated"
        and bool(deployment_status.get("can_enter_main_eval")) is True
        and checksum["status"] == "ok"
        and bool(card_contract)
    )
    metadata_keys = sorted({key for split in SPLITS for key in split_scans[split]["metadata_keys"]})
    input_types = Counter()
    target_types = Counter()
    for split in SPLITS:
        input_types.update(split_scans[split]["input_type_counts"])
        target_types.update(split_scans[split]["target_type_counts"])
    row = {
        "task": task,
        "version_path": str(version_dir),
        "dataset_card_status": dataset_card.get("status"),
        "dataset_card_can_enter_main_eval": dataset_card.get("can_enter_main_eval"),
        "deployment_status": deployment_status.get("status"),
        "deployment_can_enter_main_eval": deployment_status.get("can_enter_main_eval"),
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "non_causal_contract_present": bool(card_contract),
        "checksum_status": checksum["status"],
        "train_rows": split_scans["train"]["rows"],
        "validation_rows": split_scans["validation"]["rows"],
        "test_rows": split_scans["test"]["rows"],
        "input_schema": json.dumps(dict(input_types), sort_keys=True, ensure_ascii=False),
        "target_schema": json.dumps(dict(target_types), sort_keys=True, ensure_ascii=False),
        "input_length_min": input_stats["min"],
        "input_length_mean": input_stats["mean"],
        "input_length_p95": input_stats["p95"],
        "input_length_max": input_stats["max"],
        "target_length_min": target_stats["min"],
        "target_length_mean": target_stats["mean"],
        "target_length_p95": target_stats["p95"],
        "target_length_max": target_stats["max"],
        "token_value_estimate": max(split_scans[split]["token_value_estimate"] for split in SPLITS),
        "target_value_estimate": max(split_scans[split]["target_value_estimate"] for split in SPLITS),
        "metadata_keys": ",".join(metadata_keys),
        "recommended_metric": SELECTED_PROBES[task]["primary_metric"],
        "loss_family": SELECTED_PROBES[task]["loss_family"],
        "ruler_subtasks": json.dumps(split_scans["train"]["ruler_subtasks"], sort_keys=True, ensure_ascii=False),
        "status": "validated" if status_ok else "failed",
    }
    detail = {
        "task": task,
        "version_dir": str(version_dir),
        "dataset_card": dataset_card,
        "deployment_status": deployment_status,
        "checksums": checksum,
        "splits": {
            split: {
                key: value
                for key, value in split_scans[split].items()
                if key not in {"input_lengths", "target_lengths"}
            }
            for split in SPLITS
        },
        "row": row,
    }
    return row, detail


def write_report(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    report = Path("reports/v08_phase1_probe_data_audit_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        "| {task} | {status} | {train_rows}/{validation_rows}/{test_rows} | {input_length_min:.0f}/{input_length_mean:.1f}/{input_length_max:.0f} | {target_length_min:.0f}/{target_length_mean:.1f}/{target_length_max:.0f} | {recommended_metric} |".format(**row)
        for row in rows
    )
    report.write_text(
        f"""# v08 Phase 1 Probe Data Contract Audit 报告

## 结论

本阶段按 `ref/zigzag_experiment_execution_manual_v08.md` 只审计 6 个 `validated` 且 `can_enter_main_eval=true` 的 probe 数据版本。审计状态：`{summary['status']}`。`lra_pathfinder` 与 `lra_pathx` 未进入本阶段清单。

| task | 状态 | train/validation/test 行数 | input 长度 min/mean/max | target 长度 min/mean/max | 推荐主指标 |
|---|---|---:|---:|---:|---|
{table}

## 参数说明

| 参数名 | 中文含义 | 单位或取值 | 来源 | 记录原因 | 不适用时的处理 |
|---|---|---|---|---|---|
| attention_contract | 注意力合同 | non_causal | dataset_card.non_causal_contract 与 v08 手册 | 确认主实验不是 causal LM | 不适用时不得进入主评测 |
| causal | 是否使用 causal mask | false | v08 手册 | 防止误用 next-token 任务 | 不适用时不得进入主评测 |
| checksum_status | sha256 校验状态 | ok/failed | checksums.sha256 | 确认数据内容可追溯 | failed 时阻止后续 phase |
| train_rows | 训练集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| validation_rows | 验证集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| test_rows | 测试集行数 | 样本数 | JSONL 行数 | 和部署合同核对 | 无 |
| input_schema | input 字段类型分布 | JSON 字符串 | JSONL 扫描 | 决定编码器和模型输入 | 无 |
| target_schema | target 字段类型分布 | JSON 字符串 | JSONL 扫描 | 决定 loss 和 metric | 无 |
| input_length_min/mean/max | 输入长度统计 | token/字符合同长度 | JSONL 与 metadata | Phase 4 选择长度和图规模 | 无 |
| target_length_min/mean/max | 目标长度统计 | token/标签数 | JSONL 与 metadata | Phase 4 选择 readout 和 loss | 无 |
| recommended_metric | 推荐主指标 | metric 名称 | v08 手册和数据 schema | Phase 4 冻结主指标 | schema 不匹配时在 Phase 4 修正 |

## 报告审计

| 字段 | 值 |
|---|---|
| report_language | zh |
| explained_parameter_count | 13 |
| unexplained_parameters | [] |
| english_only_sections | [] |
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probes_v08_data_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    sample_preview: list[dict] = []
    for task in SELECTED_PROBES:
        row, detail = audit_task(task, args.output_dir, sample_preview)
        rows.append(row)
        details.append(detail)
    checksums = {detail["task"]: detail["checksums"] for detail in details}
    summary = {
        "version": EXPERIMENT_VERSION,
        "phase": "phase1_probe_data_contract_audit",
        "timestamp_utc": utc_now(),
        "status": "validated" if all(row["status"] == "validated" for row in rows) else "failed",
        "task_count": len(rows),
        "validated_task_count": sum(1 for row in rows if row["status"] == "validated"),
        "forbidden_tasks_excluded": ["lra_pathfinder", "lra_pathx"],
        "attention_contract": ATTENTION_CONTRACT,
        "causal": False,
        "command": command_string(),
        "rows": rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "task_audit.csv", rows)
    write_jsonl(args.output_dir / "task_audit.jsonl", rows)
    write_json(args.output_dir / "checksums_verification.json", checksums)
    write_jsonl(args.output_dir / "sample_preview.jsonl", sample_preview)
    write_json(args.output_dir / "task_audit_details.json", details)
    write_command(args.output_dir / "command.sh")
    write_report(args.output_dir, summary, rows)
    if summary["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
