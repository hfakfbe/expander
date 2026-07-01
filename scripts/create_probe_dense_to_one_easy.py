from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from graph_diagnostics import certificate_for_artifact
from graph_structures import build_graph_artifact


ROOT = Path(".")
BASE_MANIFEST = ROOT / "configs/probes_corrected_valid_as_test_l8_log5_task_parameters.json"
DATA_ROOT = ROOT / "datasets/probes_dense_to_one_easy_v02"
OUTPUT_ROOT = ROOT / "outputs/probes_dense_to_one_easy_v02"
CONFIG_ROOT = ROOT / "configs"
TASKS = ["selective_copy", "induction_associative_recall", "lra_listops"]
ROWS_BY_TASK = {
    "selective_copy": 256,
    "induction_associative_recall": 256,
    "lra_listops": 400,
}
BLOCK_SIZE = 16
GRAPH_DEGREE = 2

SEQ_LEN = {
    "selective_copy": 64,
    "induction_associative_recall": 64,
    "lra_listops": 64,
}
TARGET_LEN = {
    "selective_copy": 4,
    "induction_associative_recall": 4,
    "lra_listops": 1,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_records() -> dict[str, dict[str, Any]]:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    return {row["task"]: row for row in manifest["tasks"] if row["task"] in TASKS}


def value_combos(width: int, values: list[int]) -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = [()]
    for _ in range(width):
        combos = [prefix + (value,) for prefix in combos for value in values]
    return combos


def selective_rows(split: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_len = SEQ_LEN["selective_copy"]
    target_len = TARGET_LEN["selective_copy"]
    source_positions = [4, 18, 32, 46]
    marker = 15
    combos = value_combos(target_len, [1, 2, 3, 4])
    if split == "test":
        combos = list(reversed(combos))
    for index in range(count):
        values = list(combos[index % len(combos)])
        tokens = [0] * (input_len - target_len) + [marker] * target_len
        for pos, value in zip(source_positions, values):
            tokens[pos] = value
        rows.append(
            {
                "id": f"selective_copy.dense_to_one_easy_v02.{split}.{index}",
                "input": tokens,
                "metadata": {
                    "input_length": input_len,
                    "l_memorize": target_len,
                    "l_noise": input_len - target_len,
                    "seed": index,
                    "source": "dense_to_one_easy_v02",
                    "source_positions": source_positions,
                    "split": split,
                    "target_length": target_len,
                },
                "target": values,
                "task": "selective_copy",
                "variant": "dense_to_one_easy_v02",
            }
        )
    return rows


def induction_rows(split: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_len = SEQ_LEN["induction_associative_recall"]
    pair_count = TARGET_LEN["induction_associative_recall"]
    key_positions = [0, 1, 2, 3]
    value_positions = [16, 17, 18, 19]
    query_positions = [32, 33, 34, 35]
    target_positions = [48, 49, 50, 51]
    keys = [10, 11, 12, 13]
    values = [30, 31, 32, 33]
    orders = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ]
    if split == "test":
        orders = list(reversed(orders))
    for index in range(count):
        order = orders[index % len(orders)]
        tokens = [0] * input_len
        for slot, (key, value) in enumerate(zip(keys, values)):
            tokens[key_positions[slot]] = key
            tokens[value_positions[slot]] = value
        targets = []
        for query_slot, pair_index in enumerate(order):
            tokens[query_positions[query_slot]] = keys[pair_index]
            tokens[target_positions[query_slot]] = 0
            targets.append({"position": target_positions[query_slot], "value": values[pair_index]})
        rows.append(
            {
                "id": f"induction_associative_recall.dense_to_one_easy_v02.{split}.{index}",
                "input": tokens,
                "metadata": {
                    "input_length": input_len,
                    "input_seq_len": input_len,
                    "key_positions": key_positions,
                    "num_kv_pairs": pair_count,
                    "query_positions": query_positions,
                    "seed": index,
                    "source": "dense_to_one_easy_v02",
                    "split": split,
                    "target_length": pair_count,
                    "target_positions": target_positions,
                    "value_positions": value_positions,
                    "vocab_size": 8192,
                },
                "target": targets,
                "task": "induction_associative_recall",
                "variant": "dense_to_one_easy_v02",
            }
        )
    return rows


def listops_label(op: str, a: int, b: int) -> int:
    if op == "[MAX":
        return max(a, b)
    if op == "[MIN":
        return min(a, b)
    if op == "[SM":
        return (a + b) % 10
    if op == "[MED":
        return sorted([a, b, (a + b) % 10])[1]
    raise ValueError(op)


def listops_rows(split: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ops = ["[MAX", "[MIN", "[SM", "[MED"]
    examples = [(op, a, b) for op in ops for a in range(10) for b in range(10)]
    if split == "test":
        examples = list(reversed(examples))
    for index in range(count):
        op, a, b = examples[index % len(examples)]
        label = listops_label(op, a, b)
        tokens = ["(", op, str(a), str(b), ")", "]"]
        rows.append(
            {
                "id": f"lra_listops.dense_to_one_easy_v02.{split}.{index}",
                "input": tokens,
                "metadata": {
                    "input_length": len(tokens),
                    "max_args": 2,
                    "max_depth": 1,
                    "seed": index,
                    "source": "dense_to_one_easy_v02",
                    "split": split,
                    "target_length": 1,
                    "tree_length": len(tokens),
                },
                "target": label,
                "task": "lra_listops",
                "variant": "dense_to_one_easy_v02",
            }
        )
    return rows


def task_rows(task: str, split: str, count: int) -> list[dict[str, Any]]:
    if task == "selective_copy":
        return selective_rows(split, count)
    if task == "induction_associative_recall":
        return induction_rows(split, count)
    if task == "lra_listops":
        return listops_rows(split, count)
    raise ValueError(task)


def write_dataset(task: str, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, str]:
    task_dir = DATA_ROOT / task
    write_jsonl(task_dir / "train.jsonl", train_rows)
    write_jsonl(task_dir / "test.jsonl", test_rows)
    card = {
        "dataset": task,
        "split_policy": "finite_language_coverage_train_and_test_rows_for_dense_to_one_calibration",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "variant": "dense_to_one_easy_v02",
    }
    write_json(task_dir / "dataset_card.json", card)
    (task_dir / "source.lock").write_text("source=dense_to_one_easy_v02\n", encoding="utf-8")
    controlled = ["train.jsonl", "test.jsonl", "dataset_card.json", "source.lock"]
    checksums = [f"{sha256_file(task_dir / name)}  {name}" for name in controlled]
    (task_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return {
        "version_path": str(task_dir),
        "train_sha256": sha256_file(task_dir / "train.jsonl"),
        "test_sha256": sha256_file(task_dir / "test.jsonl"),
        "dataset_card_sha256": sha256_file(task_dir / "dataset_card.json"),
        "source_lock_sha256": sha256_file(task_dir / "source.lock"),
        "checksums_sha256": sha256_file(task_dir / "checksums.sha256"),
    }


def write_graph(task: str, seq_len: int) -> dict[str, Any]:
    graph_dir = OUTPUT_ROOT / "graphs" / task
    graph_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_graph_artifact(
        N_task=seq_len,
        T_raw=seq_len,
        block_size=BLOCK_SIZE,
        degree=GRAPH_DEGREE,
        graph_seed=0,
        g_config={"max_parallel_edges_per_block_pair": None},
        version="probes_dense_to_one_easy_v02",
    )
    artifact["allow_multiedges"] = True
    artifact["preserve_multiplicity"] = True
    artifact["graph_generation_algorithm"] = "dense_to_one_easy_v02_graph"
    cert = certificate_for_artifact(
        artifact,
        {"acceptance": {"rho_bound_lt": 1.0, "max_remote_local_overlap_mean": 1.0}},
    )
    artifact["certificate"] = cert
    selected = graph_dir / "selected_graph.json"
    certificate = graph_dir / "graph_certificate.json"
    generation = graph_dir / "graph_generation.json"
    write_json(selected, artifact)
    write_json(certificate, cert)
    sha = sha256_file(selected)
    (graph_dir / "graph_artifact.sha256").write_text(sha + "  selected_graph.json\n", encoding="utf-8")
    write_json(
        generation,
        {
            "status": "ok",
            "graph_generation_algorithm": "dense_to_one_easy_v02_graph",
            "graph_seed": 0,
            "N_task": seq_len,
            "T_raw": seq_len,
            "T": artifact["T"],
            "q": artifact["q"],
            "B": BLOCK_SIZE,
            "d": GRAPH_DEGREE,
            "selected_graph_path": str(selected),
            "graph_certificate_path": str(certificate),
            "canonical_graph_artifact_sha256": sha,
        },
    )
    return {
        "artifact": artifact,
        "certificate": cert,
        "graph_artifact_sha256_path": str(graph_dir / "graph_artifact.sha256"),
        "graph_certificate_path": str(certificate),
        "graph_dir": str(graph_dir),
        "graph_generation_path": str(generation),
        "selected_graph_path": str(selected),
        "selected_graph_sha256": sha,
    }


def length_stats(rows: list[dict[str, Any]], classification: bool) -> dict[str, float | int]:
    input_lengths = [len(row["input"]) for row in rows]
    if classification:
        target_lengths = [1 for _row in rows]
    else:
        target_lengths = [len(row["target"]) for row in rows]
    return {
        "input_min": min(input_lengths),
        "input_mean": sum(input_lengths) / len(input_lengths),
        "input_p95": sorted(input_lengths)[math.ceil(0.95 * len(input_lengths)) - 1],
        "input_max": max(input_lengths),
        "target_min": min(target_lengths),
        "target_mean": sum(target_lengths) / len(target_lengths),
        "target_p95": sorted(target_lengths)[math.ceil(0.95 * len(target_lengths)) - 1],
        "target_max": max(target_lengths),
    }


def update_record(
    task: str,
    base: dict[str, Any],
    data: dict[str, str],
    graph: dict[str, Any],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record = dict(base)
    classification = task == "lra_listops"
    all_rows = train_rows + test_rows
    stats = length_stats(all_rows, classification)
    seq_len = SEQ_LEN[task]
    target_len = TARGET_LEN[task]
    record.update(
        {
            "version": "probes_dense_to_one_easy_v02",
            "version_path": data["version_path"],
            "valid_as_test_v01": False,
            "dataset_source": "dense_to_one_easy_v02_generated",
            "dataset_revision_or_hash": "local_generated_v02",
            "dataset_card_sha256": data["dataset_card_sha256"],
            "source_lock_sha256": data["source_lock_sha256"],
            "checksums_sha256": data["checksums_sha256"],
            "source_train_sha256": data["train_sha256"],
            "source_validation_sha256_used_as_test": data["test_sha256"],
            "resolved_train_split_sha256": data["train_sha256"],
            "resolved_test_split_sha256": data["test_sha256"],
            "resolved_validation_split_sha256": "not_applicable",
            "train_content_sha256": data["train_sha256"],
            "test_content_sha256": data["test_sha256"],
            "discarded_old_test_sha256": "not_applicable",
            "resolved_train_examples": len(train_rows),
            "resolved_test_examples": len(test_rows),
            "resolved_validation_examples": 0,
            "resolved_runtime_input_length": seq_len,
            "resolved_runtime_target_length": target_len,
            "resolved_raw_sequence_length": seq_len,
            "resolved_padded_sequence_length": seq_len,
            "runtime_padding_positions": 0,
            "runtime_padding_policy": "fixed_block_aligned_dense_to_one_easy_v02",
            "resolved_readout_start": seq_len - target_len if task == "selective_copy" else 0,
            "resolved_sequence_length_min": stats["input_min"],
            "resolved_sequence_length_mean": stats["input_mean"],
            "resolved_sequence_length_p95": stats["input_p95"],
            "resolved_sequence_length_max": stats["input_max"],
            "resolved_target_length_min": stats["target_min"],
            "resolved_target_length_mean": stats["target_mean"],
            "resolved_target_length_p95": stats["target_p95"],
            "resolved_target_length_max": stats["target_max"],
            "resolved_layers": 8,
            "resolved_d_model": 128,
            "resolved_heads": 4,
            "resolved_ffn_dim": 512,
            "resolved_dropout": 0.0,
            "resolved_batch_size": 8,
            "resolved_gradient_accumulation_steps": 2,
            "resolved_effective_batch_size": 16,
            "resolved_eval_batch_size": 16,
            "resolved_base_learning_rate": 3.0e-4,
            "resolved_min_learning_rate": 3.0e-5,
            "resolved_lr_scheduler": "cosine",
            "resolved_warmup_ratio": 0.0,
            "resolved_weight_decay": 0.0,
            "resolved_grad_clip_norm": 1.0,
            "resolved_attention_backend": "auto_split",
            "position_encoding": "rope",
            "rope_theta": 10000.0,
            "resolved_graph_block_size": BLOCK_SIZE,
            "resolved_graph_degree_or_budget": GRAPH_DEGREE,
            "resolved_graph_generation_algorithm": "dense_to_one_easy_v02_graph",
            "resolved_graph_id": f"dense_to_one_easy_v02_{task}_B{BLOCK_SIZE}_d{GRAPH_DEGREE}_s0",
            "resolved_graph_num_blocks_or_nodes": seq_len // BLOCK_SIZE,
            "resolved_graph_seed": 0,
            "resolved_q_alias_if_applicable": seq_len // BLOCK_SIZE,
            "graph_artifacts": graph,
            "resolved_required_methods": ["dense", "random_regular", "random_memory"],
            "resolved_optional_methods": [],
            "main": {
                "checkpoint_every": 0,
                "epochs": 120,
                "log_every": 100,
                "steps": 3000,
                "test_examples": len(test_rows),
                "train_diagnostic_examples": min(128, len(train_rows)),
            },
            "smoke": {
                "checkpoint_every": 0,
                "epochs": 1,
                "log_every": 1,
                "steps": 2,
                "test_examples": min(16, len(test_rows)),
                "train_diagnostic_examples": min(16, len(train_rows)),
            },
        }
    )
    record["input_contract"] = dict(record["input_contract"])
    record["input_contract"].update(
        {
            "input_length_policy": f"dense_to_one_easy_v02_fixed_T_{seq_len}",
            "source_test_policy": "generated_finite_language_coverage_test_rows",
            "test_source": "generated_test_jsonl",
        }
    )
    return record


def main() -> None:
    base_records = selected_records()
    records = []
    for task in TASKS:
        train_rows = task_rows(task, "train", ROWS_BY_TASK[task])
        test_rows = task_rows(task, "test", ROWS_BY_TASK[task])
        data = write_dataset(task, train_rows, test_rows)
        graph = write_graph(task, SEQ_LEN[task])
        records.append(update_record(task, base_records[task], data, graph, train_rows, test_rows))

    manifest = {
        "branch_name": "codex/probes-dense-to-one-easy-v02",
        "phase": "probes_dense_to_one_easy_v02",
        "version": "probes_dense_to_one_easy_v02",
        "tasks": records,
    }
    manifest_path = CONFIG_ROOT / "probes_dense_to_one_easy_v02_task_parameters.json"
    write_json(manifest_path, manifest)

    common = {
        "output_root": "outputs/probes_dense_to_one_easy_v02/runs",
        "phase": "probes_dense_to_one_easy_v02",
        "profile": "main",
        "seeds": [0],
        "task_parameter_manifest": str(manifest_path),
        "tasks": TASKS,
        "version": "probes_dense_to_one_easy_v02",
    }
    write_json(
        CONFIG_ROOT / "probes_dense_to_one_easy_v02_dense.json",
        {
            **common,
            "methods": ["dense"],
            "trial_id": "dense_to_one_easy_v02_dense",
        },
    )
    write_json(
        CONFIG_ROOT / "probes_dense_to_one_easy_v02_random_density10.json",
        {
            **common,
            "methods": ["random_regular", "random_memory"],
            "random": {
                "actual_mask_density": 0.1,
                "exclude_block_local": True,
                "use_log_m": False,
            },
            "random_memory": {
                "enabled": True,
                "edge_scope": "all",
                "head_merge": "mean",
                "lazy_alpha": 0.5,
                "scale": 2.0,
                "source": "input",
                "steps": 1,
                "update": "lazy",
                "weight_mode": "soft",
            },
            "trial_id": "dense_to_one_easy_v02_random_density10",
        },
    )
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
