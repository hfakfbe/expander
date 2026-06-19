from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graph_structures import build_graph_artifact
from probe_common import file_sha256, write_json
from probe_tasks import build_listops_encoder, integer_encoder
from synthetic_mvp_core.artifacts import build_method_counts, build_random_remote_rows_aligned_to_zigzag_noncausal


VERSION = "probes_corrected_valid_as_test_l8_log5"
BRANCH = "codex/probes-corrected-valid-as-test-l8-log5"
METHODS = ["local", "zigzag_certified", "random_regular"]
GRAPH_DEGREE = 8
GRAPH_SEED = 0

TASK_SPECS: dict[str, dict[str, Any]] = {
    "selective_copy": {
        "loss_type": "sequence_cross_entropy",
        "primary_metric": "selective_copy_token_accuracy",
        "secondary_metrics": ["selective_copy_sequence_accuracy"],
        "encoder": "integer_shift",
        "max_value": 15,
        "block_size": 32,
        "input_length_policy": "fixed_source_train_and_validation_length_4128",
        "target_position_policy": "last_target_length_positions_of_original_input_marker_tail",
        "target_length": 16,
        "d_model": 128,
        "heads": 4,
        "ffn_dim": 512,
    },
    "induction_associative_recall": {
        "loss_type": "mqar_position_cross_entropy",
        "primary_metric": "retrieval_exact_match",
        "secondary_metrics": ["retrieval_token_accuracy"],
        "encoder": "integer_shift",
        "max_value": 8191,
        "block_size": 32,
        "input_length_policy": "max_source_train_and_validation_length_256",
        "target_position_policy": "json_target_position_inside_original_input",
        "target_length": 64,
        "d_model": 128,
        "heads": 4,
        "ffn_dim": 512,
    },
    "lra_listops": {
        "loss_type": "classification_cross_entropy",
        "primary_metric": "listops_accuracy",
        "secondary_metrics": ["listops_macro_accuracy"],
        "encoder": "listops_vocab",
        "block_size": 109,
        "input_length_policy": "max_source_train_and_validation_length_5995",
        "target_position_policy": "classification_head_pooled_original_input",
        "target_length": 1,
        "d_model": 128,
        "heads": 4,
        "ffn_dim": 512,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def input_length(row: dict[str, Any]) -> int:
    value = row.get("input")
    return len(value) if hasattr(value, "__len__") else 0


def target_length(row: dict[str, Any]) -> int:
    value = row.get("target")
    if isinstance(value, (list, str)):
        return len(value)
    if value is None:
        return 0
    return 1


def stats(values: list[int]) -> dict[str, float | int]:
    values = sorted(int(value) for value in values)
    if not values:
        return {"min": 0, "mean": 0.0, "p95": 0, "max": 0}
    idx = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    return {"min": values[0], "mean": sum(values) / len(values), "p95": values[idx], "max": values[-1]}


def light_certificate(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": artifact["version"],
        "graph_id": artifact["graph_id"],
        "graph_seed": artifact["graph_seed"],
        "T_raw": artifact["T_raw"],
        "T": artifact["T"],
        "N_task": artifact["N_task"],
        "B": artifact["B"],
        "d": artifact["d"],
        "q": artifact["q"],
        "G_type": artifact["G"]["type"],
        "H_type": artifact["H"]["type"],
        "rho_exact": "skipped_for_corrected_probe_preparation",
        "rho_zigzag_exact": "skipped_for_corrected_probe_preparation",
        "rho_zigzag_certified": "not_evaluated_in_light_certificate",
        "certified": "not_evaluated_in_light_certificate",
        "certificate_policy": "structural graph artifact validation only; expensive spectral certificate not required for input-contract correction",
    }


def write_graph(task: str, raw_t: int, block_size: int, output_root: Path) -> dict[str, Any]:
    graph_dir = output_root / "graphs" / task
    graph_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_graph_artifact(
        N_task=raw_t,
        T_raw=raw_t,
        block_size=block_size,
        degree=GRAPH_DEGREE,
        graph_seed=GRAPH_SEED,
        g_config={"max_parallel_edges_per_block_pair": None},
        version=VERSION,
    )
    artifact["allow_multiedges"] = True
    artifact["preserve_multiplicity"] = True
    artifact["graph_generation_algorithm"] = "probes_corrected_valid_as_test_l8_log5_task_parameter_selection"
    cert = light_certificate(artifact)
    artifact["certificate"] = cert
    selected = graph_dir / "selected_graph.json"
    certificate = graph_dir / "graph_certificate.json"
    generation = graph_dir / "graph_generation.json"
    write_json(selected, artifact)
    write_json(certificate, cert)
    selected_sha = file_sha256(selected)
    (graph_dir / "graph_artifact.sha256").write_text(f"{selected_sha}  selected_graph.json\n", encoding="utf-8")
    write_json(
        generation,
        {
            "B": block_size,
            "N_task": raw_t,
            "T": artifact["T"],
            "T_raw": raw_t,
            "canonical_graph_artifact_sha256": selected_sha,
            "d": GRAPH_DEGREE,
            "generation_attempts": 1,
            "graph_generation_algorithm": artifact["graph_generation_algorithm"],
            "graph_seed": GRAPH_SEED,
            "q": artifact["q"],
            "selected_graph_path": str(selected),
            "status": "ok",
            "timestamp_utc": utc_now(),
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
        "selected_graph_sha256": selected_sha,
    }


def iter_set_bits(value: int):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def rows_to_bits(rows: list[Counter[int]] | None, seq_len: int) -> list[int]:
    if rows is None:
        all_bits = (1 << seq_len) - 1
        return [all_bits for _ in range(seq_len)]
    out: list[int] = []
    for counts in rows:
        bits = 0
        for dst in counts:
            bits |= 1 << int(dst)
        out.append(bits)
    return out


def representative_targets(task: str, rows: list[dict[str, Any]], target_limit: int) -> list[tuple[int, int]]:
    if task == "selective_copy":
        pairs: list[tuple[int, int]] = []
        for row in rows[:64]:
            input_values = row["input"]
            target_values = row["target"]
            source_positions = [
                idx
                for idx, value in enumerate(input_values)
                if int(value) not in {0, 15}
            ]
            if len(source_positions) != len(target_values):
                continue
            readout_start = len(input_values) - len(target_values)
            pairs.extend((readout_start + offset, source_positions[offset]) for offset in range(len(target_values)))
        return pairs[: min(len(pairs), 512)]
    if task == "induction_associative_recall":
        entries: list[tuple[int, int]] = []
        for row in rows[:64]:
            for item in row["target"]:
                entries.append((int(item["position"]), 0))
        return entries[: min(len(entries), 512)]
    return []


def shortest_path_stats(task: str, method: str, rows: list[Counter[int]] | None, layers: int, seq_len: int, targets: list[tuple[int, int]]) -> dict[str, Any]:
    if not targets:
        return {
            "task": task,
            "method": method,
            "layers": layers,
            "target_in_Lhop_rate": "not_applicable_for_classification",
            "unreachable_rate": "not_applicable_for_classification",
        }
    row_bits = rows_to_bits(rows, seq_len)
    reached_by_hop = {i: 0 for i in range(1, layers + 1)}
    hist: dict[str, int] = {str(i): 0 for i in range(1, layers + 1)}
    hist["unreachable"] = 0
    shortest: list[int] = []
    for query, key in targets:
        frontier = 1 << int(query)
        found: int | None = None
        for hop in range(1, layers + 1):
            next_bits = 0
            for pos in iter_set_bits(frontier):
                next_bits |= row_bits[pos]
            frontier = next_bits
            if (frontier >> int(key)) & 1:
                found = hop
                break
        if found is None:
            hist["unreachable"] += 1
        else:
            hist[str(found)] += 1
            shortest.append(found)
            for hop in range(found, layers + 1):
                reached_by_hop[hop] += 1
    denom = max(len(targets), 1)
    return {
        "task": task,
        "method": method,
        **{f"target_in_{hop}hop_rate": reached_by_hop[hop] / denom for hop in range(1, layers + 1)},
        "target_in_Lhop_rate": reached_by_hop[layers] / denom,
        "average_shortest_path": sum(shortest) / len(shortest) if shortest else None,
        "unreachable_rate": hist["unreachable"] / denom,
        "per_target_shortest_path_histogram": hist,
        "direction_definition": "mask[query/readout position, source key position] == true; multi-hop follows query-to-key rows backward through layers",
    }


def reachability(task: str, graph: dict[str, Any], layers: int, seq_len: int, targets: list[tuple[int, int]]) -> dict[str, Any]:
    base = SimpleNamespace(block_size=graph["artifact"]["B"], degree=GRAPH_DEGREE, graph_config=graph["artifact"], seed=GRAPH_SEED)
    out: dict[str, Any] = {}
    for method in METHODS:
        if method == "random_regular":
            args = SimpleNamespace(block_size=graph["artifact"]["B"], degree=GRAPH_DEGREE, graph_config=graph["artifact"], seed=GRAPH_SEED)
            args.random_aligned_rows = build_random_remote_rows_aligned_to_zigzag_noncausal(seq_len, args)
            rows = build_method_counts("random_regular", seq_len, args)
        else:
            rows = build_method_counts(method, seq_len, base)
        out[method] = shortest_path_stats(task, method, rows, layers, seq_len, targets)
    return out


def encoder_for(task: str, spec: dict[str, Any], data_dir: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    path = output_root / "encoders" / task / "encoder.json"
    if spec["encoder"] == "integer_shift":
        payload = integer_encoder(int(spec["max_value"]), path)
        return path, payload
    if spec["encoder"] == "listops_vocab":
        payload = build_listops_encoder(data_dir / "train.jsonl", path)
        return path, payload
    raise ValueError(f"unknown encoder spec for {task}: {spec['encoder']}")


def build_record(task: str, data_dir: Path, output_root: Path, layers: int, log_every: int) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    card = read_json(data_dir / "dataset_card.json")
    train_rows = read_jsonl(data_dir / "train.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    train_input_lengths = [input_length(row) for row in train_rows]
    test_input_lengths = [input_length(row) for row in test_rows]
    train_target_lengths = [target_length(row) for row in train_rows]
    test_target_lengths = [target_length(row) for row in test_rows]
    input_stats = stats(train_input_lengths + test_input_lengths)
    target_stats = stats(train_target_lengths + test_target_lengths)
    raw_t = int(input_stats["max"])
    block_size = int(spec["block_size"])
    if raw_t % block_size != 0:
        raise ValueError(f"{task}: raw input length {raw_t} is not divisible by block_size={block_size}")
    encoder_path, encoder_payload = encoder_for(task, spec, data_dir, output_root)
    graph = write_graph(task, raw_t, block_size, output_root)
    runtime_target_length = int(target_stats["max"])
    targets = representative_targets(task, train_rows + test_rows, runtime_target_length)
    reach_path = output_root / "graphs" / task / "reachability.json"
    write_json(reach_path, reachability(task, graph, layers, raw_t, targets))
    effective_batch = 16
    train_rows_count = int(card["stats"]["generated_train"]["rows"])
    steps = math.ceil(train_rows_count / effective_batch)
    padding_policy = (
        "none_fixed_length_inputs"
        if int(input_stats["min"]) == int(input_stats["max"])
        else "input_only_right_padding_to_max_input_length_no_target_labels_in_padding"
    )
    return {
        "version": VERSION,
        "task": task,
        "version_path": str(data_dir),
        "no_target_append_v01": True,
        "valid_as_test_v01": True,
        "input_contract_status": "trainable",
        "input_contract": {
            "no_target_append": True,
            "keep_input_length_unchanged": True,
            "target_position_policy": spec["target_position_policy"],
            "input_length_policy": spec["input_length_policy"],
            "source_test_policy": "discarded",
            "test_source": "source_validation_jsonl",
        },
        "primary_metric": spec["primary_metric"],
        "secondary_metrics": spec["secondary_metrics"],
        "resolved_loss_type": spec["loss_type"],
        "resolved_runtime_input_length": raw_t,
        "resolved_runtime_target_length": runtime_target_length,
        "resolved_readout_start": "not_applicable_no_external_readout",
        "resolved_raw_sequence_length": raw_t,
        "resolved_padded_sequence_length": raw_t,
        "runtime_padding_policy": padding_policy,
        "runtime_padding_positions": 0 if padding_policy == "none_fixed_length_inputs" else "variable_input_padding_only",
        "resolved_sequence_length_min": input_stats["min"],
        "resolved_sequence_length_mean": input_stats["mean"],
        "resolved_sequence_length_p95": input_stats["p95"],
        "resolved_sequence_length_max": input_stats["max"],
        "resolved_target_length_min": target_stats["min"],
        "resolved_target_length_mean": target_stats["mean"],
        "resolved_target_length_p95": target_stats["p95"],
        "resolved_target_length_max": target_stats["max"],
        "resolved_train_examples": train_rows_count,
        "resolved_validation_examples": 0,
        "resolved_test_examples": int(card["stats"]["source_validation_as_test"]["rows"]),
        "resolved_train_split_sha256": file_sha256(data_dir / "train.jsonl"),
        "resolved_test_split_sha256": file_sha256(data_dir / "test.jsonl"),
        "resolved_validation_split_sha256": "absent",
        "source_train_sha256": card["sha256"]["source_train_sha256"],
        "source_validation_sha256_used_as_test": card["sha256"]["source_validation_sha256"],
        "discarded_old_test_sha256": card["sha256"]["source_test_sha256_discarded"],
        "train_content_sha256": file_sha256(data_dir / "train.jsonl"),
        "test_content_sha256": file_sha256(data_dir / "test.jsonl"),
        "dataset_card_sha256": file_sha256(data_dir / "dataset_card.json"),
        "source_lock_sha256": file_sha256(data_dir / "source.lock"),
        "checksums_sha256": file_sha256(data_dir / "checksums.sha256"),
        "resolved_encoder_type": encoder_payload["encoder_type"],
        "resolved_tokenizer_or_encoder_path": str(encoder_path),
        "resolved_tokenizer_or_encoder_sha256": file_sha256(encoder_path),
        "resolved_vocab_or_value_space_size": int(encoder_payload["vocab_size"]),
        "resolved_token_output_size": int(encoder_payload["vocab_size"]),
        "label_or_value_space": "10_classes" if task == "lra_listops" else f"0..{int(encoder_payload['vocab_size']) - 1}",
        "position_encoding": "rope",
        "rope_learnable": False,
        "rope_theta": 10000.0,
        "rope_scaling": "none",
        "rope_apply_to": "q_and_k_only",
        "absolute_position_embedding": "none",
        "resolved_model_family": "probe_transformer_encoder_readout_rope_no_target_append",
        "resolved_layers": int(layers),
        "resolved_d_model": int(spec["d_model"]),
        "resolved_heads": int(spec["heads"]),
        "resolved_ffn_dim": int(spec["ffn_dim"]),
        "resolved_dropout": 0.0,
        "resolved_attention_backend": "auto_split",
        "resolved_graph_id": graph["artifact"]["graph_id"],
        "resolved_graph_seed": GRAPH_SEED,
        "resolved_graph_generation_algorithm": graph["artifact"]["graph_generation_algorithm"],
        "resolved_graph_block_size": block_size,
        "resolved_graph_num_blocks_or_nodes": raw_t // block_size,
        "resolved_graph_degree_or_budget": GRAPH_DEGREE,
        "resolved_q_alias_if_applicable": raw_t // block_size,
        "resolved_required_methods": METHODS,
        "resolved_optimizer": "adamw",
        "resolved_base_learning_rate": 0.0003,
        "resolved_min_learning_rate": 0.00003,
        "resolved_lr_scheduler": "cosine",
        "resolved_warmup_ratio": 0.0,
        "resolved_weight_decay": 0.01,
        "resolved_grad_clip_norm": 1.0,
        "resolved_batch_size": 2,
        "resolved_gradient_accumulation_steps": 8,
        "resolved_effective_batch_size": effective_batch,
        "resolved_eval_batch_size": 2,
        "resolved_main_steps_1epoch": steps,
        "model_seed": 0,
        "data_seed": 0,
        "dropout_seed_or_policy": "torch_seed_model_seed_dropout_disabled_by_default",
        "graph_artifacts": graph,
        "reachability_path": str(reach_path),
        "main": {
            "epochs": 1,
            "steps": steps,
            "log_every": int(log_every),
            "checkpoint_every": max(250, min(1000, steps)),
            "train_diagnostic_examples": 16,
            "test_examples": int(card["stats"]["source_validation_as_test"]["rows"]),
        },
        "smoke": {
            "epochs": 1,
            "steps": 2,
            "log_every": 1,
            "checkpoint_every": 0,
            "train_diagnostic_examples": 4,
            "test_examples": min(8, int(card["stats"]["source_validation_as_test"]["rows"])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/probes_corrected_valid_as_test_l8_log5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/probes_corrected_valid_as_test_l8_log5"))
    parser.add_argument("--config", type=Path, default=Path("configs/probes_corrected_valid_as_test_l8_log5.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/probes_corrected_valid_as_test_l8_log5_task_parameters.json"))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    materialization = read_json(args.data_root / "materialization_manifest.json")
    records = [build_record(task, args.data_root / task, args.output_root, args.layers, args.log_every) for task in TASK_SPECS]
    blocked = materialization.get("blocked_tasks", [])
    manifest = {
        "version": VERSION,
        "phase": "probes_corrected_valid_as_test_l8_log5_task_parameter_selection",
        "branch_name": BRANCH,
        "branch_head_commit": git_value("rev-parse", "HEAD"),
        "attention_contract": "non_causal",
        "causal": False,
        "graph_directionality": "directed",
        "layers": int(args.layers),
        "log_every": int(args.log_every),
        "input_contract": {
            "no_target_append": True,
            "keep_input_length_unchanged": True,
            "validation_split": "absent",
            "test_split": "source_validation_jsonl",
        },
        "tasks": records,
        "blocked_tasks": blocked,
        "timestamp_utc": utc_now(),
    }
    write_json(args.manifest, manifest)
    write_json(
        args.config,
        {
            "version": VERSION,
            "phase": "probes_corrected_valid_as_test_l8_log5",
            "profile": "main",
            "task_parameter_manifest": str(args.manifest),
            "output_root": str(args.output_root / "runs"),
            "trial_id": "main_l8_log5",
            "tasks": list(TASK_SPECS),
            "blocked_tasks": [row["task"] for row in blocked],
            "methods": METHODS,
            "seeds": [0],
            "train": {
                "epochs": 1,
                "log_every": int(args.log_every),
                "train_diagnostic_examples": 16,
            },
            "final_eval": {"test_split": "test_jsonl_from_source_validation"},
        },
    )
    branch_manifest = {
        "version": VERSION,
        "branch_name": BRANCH,
        "branch_head_commit": git_value("rev-parse", "HEAD"),
        "worktree_path": str(Path.cwd()),
        "dataset_materialization_script_sha256": file_sha256(Path("scripts/materialize_probes_corrected.py")),
        "prepare_script_sha256": file_sha256(Path("scripts/prepare_probes_corrected.py")),
        "config": str(args.config),
        "manifest": str(args.manifest),
        "created_at": utc_now(),
        "final_status": "prepared",
    }
    write_json(args.output_root / "branch_manifest.json", branch_manifest)
    print(json.dumps({"status": "ok", "config": str(args.config), "manifest": str(args.manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
