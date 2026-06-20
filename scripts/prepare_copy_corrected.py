from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graph_diagnostics import certificate_for_artifact
from graph_structures import build_graph_artifact
from probe_common import file_sha256, write_json
from probe_tasks import identity_integer_encoder
from synthetic_mvp_core.artifacts import (
    build_method_counts,
    build_random_remote_rows_aligned_to_zigzag_noncausal,
)


VERSION = "copy_corrected_v01_l8_log5"
BRANCH = "codex/copy-corrected-v01-l8-log5"
RAW_T = 2048
SOURCE_LENGTH = 1024
TARGET_LENGTH = 1024
MARKER_TOKEN_ID = 63
BLOCK_SIZE = 64
GRAPH_DEGREE = 8
GRAPH_Q = 32
GRAPH_SEED = 0
REQUIRED_METHODS = ["dense", "local", "zigzag_certified", "random_regular"]
OLD_TEST_SHA256 = "50de40e9b6f7c53af8a912cf0967ae1129e84028bcc7f90c14a94620d0760fac"


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_branch() -> None:
    branch = git_value("branch", "--show-current")
    if branch != BRANCH:
        raise SystemExit(f"must run on {BRANCH}, got {branch!r}")


def write_graph(output_root: Path) -> dict[str, Any]:
    if RAW_T % BLOCK_SIZE != 0:
        raise ValueError(f"RAW_T={RAW_T} must be divisible by block_size={BLOCK_SIZE}")
    if GRAPH_Q != RAW_T // BLOCK_SIZE:
        raise ValueError(f"q={GRAPH_Q} must equal RAW_T/block_size={RAW_T // BLOCK_SIZE}")
    if GRAPH_DEGREE <= 0 or GRAPH_DEGREE >= BLOCK_SIZE:
        raise ValueError(f"degree must satisfy 0 < d < B, got d={GRAPH_DEGREE}, B={BLOCK_SIZE}")
    graph_dir = output_root / "graphs" / "copy"
    graph_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_graph_artifact(
        N_task=RAW_T,
        T_raw=RAW_T,
        block_size=BLOCK_SIZE,
        degree=GRAPH_DEGREE,
        graph_seed=GRAPH_SEED,
        g_config={"max_parallel_edges_per_block_pair": None},
        version=VERSION,
    )
    artifact["allow_multiedges"] = True
    artifact["preserve_multiplicity"] = True
    artifact["graph_generation_algorithm"] = "copy_corrected_v01_task_parameter_selection"
    cert = certificate_for_artifact(
        artifact,
        {"acceptance": {"rho_bound_lt": 1.0, "max_remote_local_overlap_mean": 0.5}},
    )
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
            "B": BLOCK_SIZE,
            "N_task": RAW_T,
            "T": int(artifact["T"]),
            "T_raw": RAW_T,
            "canonical_graph_artifact_sha256": selected_sha,
            "command": "python scripts/prepare_copy_corrected.py",
            "d": GRAPH_DEGREE,
            "generation_attempts": 1,
            "graph_generation_algorithm": artifact["graph_generation_algorithm"],
            "graph_seed": GRAPH_SEED,
            "q": GRAPH_Q,
            "selected_graph_path": str(selected),
            "status": "ok",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
    out = []
    for counts in rows:
        bits = 0
        for dst in counts.keys():
            bits |= 1 << int(dst)
        out.append(bits)
    return out


def shortest_path_stats(method: str, rows: list[Counter[int]] | None, layers: int, seq_len: int) -> dict[str, Any]:
    row_bits = rows_to_bits(rows, seq_len)
    hist: dict[str, int] = {str(i): 0 for i in range(1, layers + 1)}
    hist["unreachable"] = 0
    reached_by_hop = {i: 0 for i in range(1, layers + 1)}
    shortest_values: list[int] = []
    for i in range(TARGET_LENGTH):
        source = i
        marker = SOURCE_LENGTH + i
        frontier = 1 << marker
        found: int | None = None
        for hop in range(1, layers + 1):
            next_bits = 0
            for pos in iter_set_bits(frontier):
                next_bits |= row_bits[pos]
            frontier = next_bits
            if (frontier >> source) & 1:
                found = hop
                break
        if found is None:
            hist["unreachable"] += 1
        else:
            hist[str(found)] += 1
            shortest_values.append(found)
            for hop in range(found, layers + 1):
                reached_by_hop[hop] += 1
    return {
        **{f"target_in_{hop}hop_rate": reached_by_hop[hop] / TARGET_LENGTH for hop in range(1, layers + 1)},
        "target_in_Lhop_rate": reached_by_hop[layers] / TARGET_LENGTH,
        "average_shortest_path": (sum(shortest_values) / len(shortest_values)) if shortest_values else None,
        "unreachable_rate": hist["unreachable"] / TARGET_LENGTH,
        "per_target_shortest_path_histogram": hist,
        "direction_definition": "mask[marker_query_position, source_key_position] == true; multi-hop follows query-to-key rows backward through layers",
        "method": method,
        "layers": layers,
    }


def reachability(graph: dict[str, Any], layers: int) -> dict[str, Any]:
    base = SimpleNamespace(
        block_size=BLOCK_SIZE,
        degree=GRAPH_DEGREE,
        graph_config=graph["artifact"],
        seed=GRAPH_SEED,
    )
    out: dict[str, Any] = {}
    for method in ["dense", "local", "zigzag_certified"]:
        rows = None if method == "dense" else build_method_counts(method, RAW_T, base)
        out[method] = shortest_path_stats(method, rows, layers, RAW_T)
    random_args = SimpleNamespace(
        block_size=BLOCK_SIZE,
        degree=GRAPH_DEGREE,
        graph_config=graph["artifact"],
        seed=GRAPH_SEED,
    )
    random_args.random_aligned_rows = build_random_remote_rows_aligned_to_zigzag_noncausal(RAW_T, random_args)
    random_rows = build_method_counts("random_regular", RAW_T, random_args)
    out["random_regular"] = shortest_path_stats("random_regular", random_rows, layers, RAW_T)
    zigzag_rows = build_method_counts("zigzag_certified", RAW_T, base)
    zigzag_k = [len(row) for row in zigzag_rows or []]
    random_k = [len(row) for row in random_rows or []]
    errors = [abs(a - b) for a, b in zip(zigzag_k, random_k)]
    out["random_regular"]["random_k_alignment_error_max"] = max(errors) if errors else 0
    out["random_regular"]["random_k_alignment_error_mean"] = sum(errors) / max(len(errors), 1)
    return out


def build_record(data_dir: Path, output_root: Path, graph: dict[str, Any], encoder_path: Path, reachability_path: Path, layers: int) -> dict[str, Any]:
    card = read_json(data_dir / "dataset_card.json")
    return {
        "copy_corrected_v01": True,
        "copy_corrected_variant": VERSION,
        "task": "copy",
        "version_path": str(data_dir),
        "primary_metric": "copy_token_accuracy",
        "secondary_metrics": ["copy_sequence_accuracy"],
        "resolved_loss_type": "sequence_cross_entropy",
        "resolved_runtime_input_length": RAW_T,
        "resolved_runtime_target_length": TARGET_LENGTH,
        "resolved_readout_start": SOURCE_LENGTH,
        "resolved_raw_sequence_length": RAW_T,
        "resolved_padded_sequence_length": RAW_T,
        "resolved_train_examples": int(card["rows"]["train"]),
        "resolved_validation_examples": 0,
        "resolved_test_examples": int(card["rows"]["test"]),
        "resolved_train_split_sha256": file_sha256(data_dir / "train.jsonl"),
        "resolved_test_split_sha256": file_sha256(data_dir / "test.jsonl"),
        "train_content_sha256": card["generated_sha256"]["train_content_sha256"],
        "test_content_sha256": card["generated_sha256"]["test_content_sha256"],
        "discarded_old_test_sha256": OLD_TEST_SHA256,
        "resolved_encoder_type": "identity_integer",
        "resolved_tokenizer_or_encoder_path": str(encoder_path),
        "resolved_tokenizer_or_encoder_sha256": file_sha256(encoder_path),
        "resolved_vocab_or_value_space_size": 64,
        "resolved_token_output_size": 64,
        "marker_token_id": MARKER_TOKEN_ID,
        "target_value_min": 1,
        "target_value_max": 62,
        "tensor_padding": "none",
        "position_encoding": "rope",
        "rope_learnable": False,
        "rope_theta": 10000.0,
        "rope_scaling": "none",
        "rope_apply_to": "q_and_k_only",
        "absolute_position_embedding": "none",
        "resolved_model_family": "probe_transformer_encoder_readout_rope_marker",
        "resolved_layers": int(layers),
        "resolved_d_model": 128,
        "resolved_heads": 4,
        "resolved_ffn_dim": 512,
        "resolved_dropout": 0.0,
        "resolved_attention_backend": "auto_split",
        "resolved_graph_id": graph["artifact"]["graph_id"],
        "resolved_graph_seed": GRAPH_SEED,
        "resolved_graph_generation_algorithm": "copy_corrected_v01_task_parameter_selection",
        "resolved_graph_block_size": BLOCK_SIZE,
        "resolved_graph_num_blocks_or_nodes": GRAPH_Q,
        "resolved_graph_degree_or_budget": GRAPH_DEGREE,
        "resolved_q_alias_if_applicable": GRAPH_Q,
        "resolved_required_methods": REQUIRED_METHODS,
        "resolved_optimizer": "adamw",
        "resolved_base_learning_rate": 0.0003,
        "resolved_min_learning_rate": 0.00003,
        "resolved_lr_scheduler": "cosine",
        "resolved_warmup_ratio": 0.0,
        "resolved_weight_decay": 0.01,
        "resolved_grad_clip_norm": 1.0,
        "resolved_batch_size": 2,
        "resolved_gradient_accumulation_steps": 8,
        "resolved_effective_batch_size": 16,
        "resolved_eval_batch_size": 2,
        "model_seed": 0,
        "data_seed": 0,
        "dropout_seed_or_policy": "torch_seed_model_seed_dropout_disabled_by_default",
        "graph_artifacts": graph,
        "reachability_path": str(reachability_path),
        "dataset_card_sha256": file_sha256(data_dir / "dataset_card.json"),
        "source_lock_sha256": file_sha256(data_dir / "source.lock"),
        "checksums_sha256": file_sha256(data_dir / "checksums.sha256"),
    }


def main() -> None:
    global VERSION, BRANCH, BLOCK_SIZE, GRAPH_DEGREE, GRAPH_Q, GRAPH_SEED, REQUIRED_METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/copy"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/copy_corrected_v01_l8_log5"))
    parser.add_argument("--config", type=Path, default=Path("configs/copy_corrected_v01_l8_log5.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/copy_corrected_v01_l8_log5_task_parameters.json"))
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--q", type=int, default=GRAPH_Q)
    parser.add_argument("--degree", type=int, default=GRAPH_DEGREE)
    parser.add_argument("--graph-seed", type=int, default=GRAPH_SEED)
    parser.add_argument("--methods", nargs="+", default=REQUIRED_METHODS)
    parser.add_argument("--trial-id", default="gate")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--branch-name", default=BRANCH)
    parser.add_argument("--skip-branch-check", action="store_true")
    args = parser.parse_args()
    VERSION = str(args.version)
    BRANCH = str(args.branch_name)
    BLOCK_SIZE = int(args.block_size)
    GRAPH_DEGREE = int(args.degree)
    GRAPH_Q = int(args.q)
    GRAPH_SEED = int(args.graph_seed)
    REQUIRED_METHODS = [str(method) for method in args.methods]
    if not args.skip_branch_check:
        require_branch()
    if not (args.data_dir / "train.jsonl").exists() or not (args.data_dir / "test.jsonl").exists():
        raise SystemExit("datasets/copy/train.jsonl and test.jsonl must exist; run scripts/materialize_copy_corrected.py first")
    if (args.data_dir / "validation.jsonl").exists():
        raise SystemExit("datasets/copy/validation.jsonl is forbidden")
    args.output_root.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_root / "encoders" / "copy" / "encoder.json"
    identity_integer_encoder(64, encoder_path)
    graph = write_graph(args.output_root)
    reach = reachability(graph, args.layers)
    reach_path = args.output_root / "graphs" / "copy" / "reachability.json"
    write_json(reach_path, reach)
    record = build_record(args.data_dir, args.output_root, graph, encoder_path, reach_path, args.layers)
    manifest = {
        "attention_contract": "non_causal",
        "base_branch": "main",
        "branch_name": BRANCH,
        "causal": False,
        "graph_directionality": "directed",
        "merge_back_to_main": False,
        "phase": "copy_corrected_v01_task_parameter_selection",
        "tasks": [record],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
    }
    write_json(args.manifest, manifest)
    write_json(
        args.config,
        {
            "version": VERSION,
            "phase": "copy_corrected_v01",
            "profile": "gate",
            "task_parameter_manifest": str(args.manifest),
            "output_root": str(args.output_root / "runs"),
            "trial_id": str(args.trial_id),
            "tasks": ["copy"],
            "methods": REQUIRED_METHODS,
            "seeds": [0],
            "train": {
                "epochs": 1,
                "log_every": int(args.log_every),
                "checkpoint_every": 100,
                "train_diagnostic_examples": 16,
            },
            "gate_overfit": {
                "method": "dense",
                "examples": 2,
                "max_steps": 1000,
                "layers": int(args.layers),
                "d_model": 128,
                "heads": 4,
                "ffn_dim": 512,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "threshold_token_accuracy": 0.999,
                "threshold_sequence_accuracy": 0.99,
                "threshold_loss": 0.01,
            },
            "final_eval": {"test_examples": 1000},
            "graph_override": {
                "q": GRAPH_Q,
                "B": BLOCK_SIZE,
                "d": GRAPH_DEGREE,
                "graph_seed": GRAPH_SEED,
            },
        },
    )
    branch_manifest_path = args.output_root / "branch_manifest.json"
    if branch_manifest_path.exists():
        branch_manifest = read_json(branch_manifest_path)
        branch_manifest.update(
            {
                "branch_head_commit": git_value("rev-parse", "HEAD"),
                "dataset_materialization_script_sha256": file_sha256(Path("scripts/materialize_copy_corrected.py")),
                "train_sha256": record["resolved_train_split_sha256"],
                "test_sha256": record["resolved_test_split_sha256"],
                "train_content_sha256": record["train_content_sha256"],
                "test_content_sha256": record["test_content_sha256"],
            }
        )
        write_json(branch_manifest_path, branch_manifest)
    print(json.dumps({"status": "ok", "config": str(args.config), "manifest": str(args.manifest), "reachability": str(reach_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
