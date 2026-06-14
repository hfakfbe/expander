from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from graph_structures import (
    DEFAULT_GRAPH_CONFIG,
    canonical_method,
    load_graph_artifact,
    validate_graph_config,
)
from v07_artifacts import materialize_graph_artifact

from .data import canonical_task_name, copy_source_length_from_total, padded_copy_lengths


DEFAULT_CONFIG = {
    "version": "v06",
    "task": {
        "name": "copy",
        "data": "online",
        "mode": "full_copy",
        "num_values": 4,
        "special_tokens": {
            "pad": 0,
            "sep": 5,
            "eos": 6,
        },
        "train_lengths": [512],
        "eval_lengths": [512],
    },
    "model": {
        "architecture": "transformer",
        "layers": 8,
        "d_model": 128,
        "heads": 4,
        "ffn_dim": 256,
        "dropout": 0.1,
        "attention_backend": "auto_split",
    },
    "attention": {
        "methods": ["dense", "local", "random_regular", "zigzag_certified"],
        "causal": True,
        "block_size": 16,
        "degree": 4,
        "graph_artifact": "outputs/copy_v06_graph_search/selected_graph.json",
        "graph": copy.deepcopy(DEFAULT_GRAPH_CONFIG),
        "multiplicity": {
            "mode": "unique_log_m",
            "boolean_ablation": True,
        },
    },
    "train": {
        "steps": 200,
        "batch_size": 8,
        "eval_batches": 5,
        "learning_rate": 1e-3,
        "seeds": [0],
        "optimizer": "adamw",
        "log_every": 10,
        "eval_every": 10,
    },
    "output": {
        "root": "outputs/synthetic_mvp",
        "plot_curves": True,
        "curve_format": "png",
    },
}

def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def file_sha256(path: Path | None) -> str:
    if path is None:
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_commit() -> str:
    env_commit = (
        os.environ.get("COPY_V06_GIT_COMMIT")
        or os.environ.get("COPY_V05_GIT_COMMIT")
        or os.environ.get("GIT_COMMIT")
    )
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""

def detect_location() -> str:
    cwd = str(Path.cwd())
    if cwd.startswith("/home/huiwei/ysx/zigzag_attention"):
        return "remote"
    if cwd.startswith("/Users/sxye/Documents/expander"):
        return "local"
    return "unknown"

def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return jsonable(value.as_dict())
    if isinstance(value, SimpleNamespace):
        return {key: jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value

def serialize_args(args) -> dict:
    return jsonable(vars(args))

def shell_command() -> str:
    return shlex.join([sys.executable, *sys.argv])

def write_command_script(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    commit = (
        os.environ.get("COPY_V06_GIT_COMMIT")
        or os.environ.get("COPY_V05_GIT_COMMIT")
        or os.environ.get("GIT_COMMIT", "")
    )
    pythonpath = os.environ.get("PYTHONPATH", "")
    env_parts = [
        f"COPY_V06_GIT_COMMIT={shlex.quote(commit)}",
        f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda)}",
    ]
    if pythonpath:
        env_parts.append(f"PYTHONPATH={shlex.quote(pythonpath)}")
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(Path.cwd()))}",
                "conda activate ysx_base 2>/dev/null || true",
                f"{' '.join(env_parts)} {command}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)

def load_config(path: Path | None) -> tuple[dict, str, str]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG), "", ""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, loaded), str(path), file_sha256(path)

def apply_cli_overrides(config: dict, cli) -> dict:
    config = copy.deepcopy(config)
    if cli.task is not None:
        config["task"]["name"] = cli.task
    if cli.seq_len is not None:
        config["task"]["train_lengths"] = [cli.seq_len]
        config["task"]["eval_lengths"] = [cli.seq_len]
    if cli.train_lengths is not None:
        config["task"]["train_lengths"] = parse_csv_ints(cli.train_lengths)
    if cli.eval_lengths is not None:
        config["task"]["eval_lengths"] = parse_csv_ints(cli.eval_lengths)
    if cli.methods is not None:
        config["attention"]["methods"] = parse_csv_strings(cli.methods)
    if cli.block_size is not None:
        config["attention"]["block_size"] = cli.block_size
    if cli.degree is not None:
        config["attention"]["degree"] = cli.degree
    if cli.steps is not None:
        config["train"]["steps"] = cli.steps
    if cli.eval_batches is not None:
        config.setdefault("eval", {})["eval_batches"] = cli.eval_batches
        config["train"]["eval_batches"] = cli.eval_batches
    if cli.batch_size is not None:
        config["train"]["batch_size"] = cli.batch_size
    if cli.learning_rate is not None:
        config["train"]["learning_rate"] = cli.learning_rate
    if cli.seed is not None:
        config["train"]["seeds"] = [cli.seed]
    if cli.seeds is not None:
        config["train"]["seeds"] = parse_csv_ints(cli.seeds)
    if cli.log_every is not None:
        config["train"]["log_every"] = cli.log_every
    if cli.eval_every is not None:
        config["train"]["eval_every"] = cli.eval_every
    if cli.d_model is not None:
        config["model"]["d_model"] = cli.d_model
    if cli.layers is not None:
        config["model"]["layers"] = cli.layers
    if cli.heads is not None:
        config["model"]["heads"] = cli.heads
    if cli.ffn_dim is not None:
        config["model"]["ffn_dim"] = cli.ffn_dim
    if cli.dropout is not None:
        config["model"]["dropout"] = cli.dropout
    if cli.attention_backend is not None:
        config["model"]["attention_backend"] = cli.attention_backend
    if cli.num_values is not None:
        config["task"]["num_values"] = cli.num_values
    if cli.num_keys is not None:
        config["task"]["num_keys"] = cli.num_keys
    if cli.output_dir is not None:
        config["output"]["root"] = str(cli.output_dir)
    return config

def build_resolved_config_snapshot(
    config: dict,
    block_size: int,
    degree: int,
    graph_artifact_path: str,
    graph_artifact: dict | None,
    graph_certificate: dict,
    padded_lengths: list[int],
    graph_materialization=None,
) -> dict:
    resolved = copy.deepcopy(config)
    attention = resolved.setdefault("attention", {})
    attention["block_size"] = int(block_size)
    attention["degree"] = int(degree)
    if graph_artifact_path:
        attention["graph_artifact"] = str(graph_artifact_path)
    if graph_materialization is not None:
        graph_cfg = resolved.setdefault("graph", {})
        graph_cfg["runtime_graph_artifact_path"] = str(graph_materialization.selected_graph_path)
        graph_cfg["runtime_graph_certificate_path"] = str(graph_materialization.certificate_path)
        graph_cfg["runtime_graph_generation_path"] = str(graph_materialization.generation_path)
        graph_cfg["graph_artifact_sha256"] = graph_materialization.graph_artifact_sha256
        graph_cfg["canonical_graph_artifact_sha256"] = graph_materialization.canonical_graph_artifact_sha256
        graph_cfg["graph_artifact_sha256_matches_canonical"] = graph_materialization.sha256_matches_canonical
    runtime_graph = {
        "graph_artifact": str(graph_artifact_path),
        "graph_id": "",
        "graph_seed": "",
        "B": int(block_size),
        "d": int(degree),
        "T": max(padded_lengths) if padded_lengths else "",
        "q": (max(padded_lengths) // int(block_size)) if padded_lengths else "",
        "graph_certified": bool(graph_certificate.get("certified", False)),
    }
    if graph_materialization is not None:
        runtime_graph.update(graph_materialization.as_dict())
    if graph_artifact is not None:
        runtime_graph.update(
            {
                "graph_id": graph_artifact.get("graph_id", ""),
                "graph_seed": graph_artifact.get("graph_seed", ""),
                "T": graph_artifact.get("T", runtime_graph["T"]),
                "q": graph_artifact.get("q", runtime_graph["q"]),
                "G_type": graph_artifact.get("G", {}).get("type", ""),
                "H_type": graph_artifact.get("H", {}).get("type", ""),
            }
        )
    attention["runtime_graph"] = runtime_graph
    return resolved

def build_runtime_args(config: dict, cli, config_path: str, config_sha: str) -> SimpleNamespace:
    user_config_snapshot = copy.deepcopy(config)
    task = config["task"]
    model = config["model"]
    attention = config["attention"]
    train = config["train"]
    eval_cfg = config.get("eval", {})
    output = config["output"]
    structure = config.get("structure", {})
    if structure:
        attention["block_size"] = int(structure.get("B", attention.get("block_size", 16)))
        attention["degree"] = int(structure.get("d", attention.get("degree", 4)))
    if "N_total" in task:
        source_length = int(task.get("copy_source_length") or copy_source_length_from_total(task["N_total"]))
        task["copy_source_length"] = source_length
        task["train_lengths"] = [source_length]
        task["eval_lengths"] = [source_length]
    train_lengths = [int(v) for v in task.get("train_lengths", task.get("sequence_lengths", [128]))]
    eval_lengths = [int(v) for v in task.get("eval_lengths", train_lengths)]
    if task.get("mode", "full_copy") != "full_copy":
        raise ValueError(f"unsupported copy mode: {task.get('mode')}")
    if model.get("architecture") != "transformer":
        raise ValueError(f"unsupported architecture: {model.get('architecture')}")
    methods = [canonical_method(str(method)) for method in attention["methods"]]
    output_dir = Path(output["root"])
    graph_materialization = None
    if config.get("graph") and not bool(config.get("graph", {}).get("generate", False)):
        graph_materialization = materialize_graph_artifact(
            config,
            output_dir,
            require=any(
                method in {"zigzag_certified", "zigzag_certified_cosine", "zigzag_boolean", "random_regular"}
                for method in methods
            ),
        )
        if graph_materialization is not None:
            attention["graph_artifact"] = str(graph_materialization.selected_graph_path)
    graph_artifact_path = attention.get("graph_artifact", "")
    graph_artifact = None
    graph_certificate = {}
    if graph_artifact_path and "<" not in str(graph_artifact_path):
        path = Path(graph_artifact_path)
        if path.exists():
            graph_artifact = load_graph_artifact(path)
            graph_certificate = dict(graph_artifact.get("certificate", {}))
        elif any(method in {"zigzag_certified", "zigzag_certified_cosine", "zigzag_boolean"} for method in methods):
            raise FileNotFoundError(f"graph_artifact not found: {path}")
    if graph_materialization is not None:
        graph_artifact = graph_materialization.artifact
        graph_certificate = graph_materialization.certificate
    if graph_artifact is not None:
        block_size = int(graph_artifact["B"])
        degree = int(graph_artifact["d"])
    else:
        block_size = int(attention.get("block_size", 16))
        degree = int(attention.get("degree", 4))
    padded_lengths = [padded_copy_lengths(N, block_size)[1] for N in [*train_lengths, *eval_lengths]]
    graph_config = graph_artifact if graph_artifact is not None else attention.get("graph", DEFAULT_GRAPH_CONFIG)
    graph_config = validate_graph_config(max(padded_lengths), block_size, degree, graph_config)
    for seq_len in padded_lengths:
        validate_graph_config(
            seq_len,
            block_size,
            degree,
            graph_config,
        )
    steps = int(train["steps"])
    log_every = int(train.get("log_every", max(1, steps // 10)))
    eval_every = int(train.get("eval_every", log_every))
    command = shell_command()
    special_tokens = task.get("special_tokens", {})
    multiplicity = attention.get("multiplicity", {})
    raw_config_snapshot = user_config_snapshot
    resolved_config_snapshot = build_resolved_config_snapshot(
        config=config,
        block_size=block_size,
        degree=degree,
        graph_artifact_path=str(graph_artifact_path),
        graph_artifact=graph_artifact,
        graph_certificate=graph_certificate,
        padded_lengths=padded_lengths,
        graph_materialization=graph_materialization,
    )
    if structure:
        expected_q = structure.get("q")
        if expected_q is not None and graph_artifact is not None and int(expected_q) != int(graph_artifact["q"]):
            raise ValueError(f"structure.q={expected_q} does not match graph q={graph_artifact['q']}")
    if str(config.get("version", "")).lower() == "v07":
        if max(padded_lengths) != 1024:
            raise ValueError(f"v07 copy expects padded T=1024, got {max(padded_lengths)}")
        if int(block_size) != 32 or int(degree) != 8:
            raise ValueError(f"v07 copy expects B=32,d=8, got B={block_size},d={degree}")
        if graph_artifact is not None and int(graph_artifact.get("q", 0)) != 32:
            raise ValueError(f"v07 copy expects q=32, got q={graph_artifact.get('q')}")
    random_alignment_mode = str(
        attention.get("random_alignment_mode")
        or config.get("attention", {}).get("random_alignment_mode")
        or "per_query"
    )
    random_target_k_source = str(
        attention.get("random_target_k_source")
        or config.get("attention", {}).get("random_target_k_source")
        or "zigzag_actual_post_causal"
    )
    method_overrides = copy.deepcopy(config.get("method_overrides", {}))
    default_lr_scheduler = str(train.get("lr_scheduler", train.get("default_lr_scheduler", "constant")))
    eval_batch_size = int(eval_cfg.get("batch_size", train.get("eval_batch_size", train["batch_size"])))
    eval_batches = eval_cfg.get("eval_batches", train.get("eval_batches", 1))
    if eval_batches == "all":
        eval_batches_value = "all"
    else:
        eval_batches_value = int(eval_batches)
    return SimpleNamespace(
        version=str(config.get("version", "v06")),
        task=canonical_task_name(task.get("name", "copy")),
        data_mode=task.get("data", "online"),
        copy_mode=task.get("mode", "full_copy"),
        num_values=int(task.get("num_values", 4)),
        N_total=int(task.get("N_total", 2 * train_lengths[0] + 2)),
        copy_source_length=int(task.get("copy_source_length", train_lengths[0])),
        pad_token=int(special_tokens.get("pad", 0)),
        sep_token=int(special_tokens.get("sep", int(task.get("num_values", 4)) + 1)),
        eos_token=int(special_tokens.get("eos", int(task.get("num_values", 4)) + 2)),
        train_lengths=train_lengths,
        eval_lengths=eval_lengths,
        methods=methods,
        block_size=block_size,
        degree=degree,
        causal=bool(attention.get("causal", True)),
        graph_config=graph_config,
        graph_artifact=graph_artifact,
        graph_artifact_path=str(graph_artifact_path),
        graph_materialization=graph_materialization,
        graph_certificate=graph_certificate,
        graph_id=str(graph_config.get("graph_id", "")),
        graph_seed=graph_config.get("graph_seed", ""),
        multiplicity_mode=str(multiplicity.get("mode", "boolean")),
        architecture=model.get("architecture", "transformer"),
        layers=int(model["layers"]),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        ffn_dim=int(model["ffn_dim"]),
        dropout=float(model["dropout"]),
        attention_backend=model["attention_backend"],
        steps=steps,
        batch_size=int(train["batch_size"]),
        eval_batch_size=eval_batch_size,
        eval_batches=eval_batches_value,
        learning_rate=float(train["learning_rate"]),
        base_learning_rate=float(train.get("base_learning_rate", train["learning_rate"])),
        lr_scheduler=default_lr_scheduler,
        method_overrides=method_overrides,
        warmup_ratio=float(train.get("warmup_ratio", 0.0)),
        min_lr_ratio=float(train.get("min_lr_ratio", 0.0)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        grad_clip_norm=float(train.get("grad_clip_norm", 0.0)),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 1)),
        effective_batch_size=int(train.get("effective_batch_size", int(train["batch_size"]))),
        random_alignment_mode=random_alignment_mode,
        random_target_k_source=random_target_k_source,
        seeds=[int(seed) for seed in train["seeds"]],
        optimizer=train.get("optimizer", "adamw").lower(),
        log_every=log_every,
        eval_every=eval_every,
        checkpoint_every=int(train.get("checkpoint_every", 0) or 0),
        output_dir=output_dir,
        device=cli.device,
        skip_tests=bool(cli.skip_tests),
        config_path=config_path,
        config_sha256=config_sha,
        raw_config_snapshot=raw_config_snapshot,
        config_snapshot=resolved_config_snapshot,
        command=command,
        log_path=cli.log_path
        or os.environ.get("COPY_V06_LOG_PATH", "")
        or os.environ.get("COPY_V05_LOG_PATH", ""),
        git_commit=git_commit(),
        local_or_remote=cli.local_or_remote or detect_location(),
        seq_len=train_lengths[0],
        seed=int(train["seeds"][0]),
        plot_curves=bool(output.get("plot_curves", True)),
        curve_format=output.get("curve_format", "png"),
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=["copy"])
    parser.add_argument("--methods")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--train-lengths")
    parser.add_argument("--eval-lengths")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--ffn-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--attention-backend",
        choices=["dense_mask", "neighbor", "split", "blockpair", "auto", "auto_split", "auto_blockpair"],
        help=(
            "dense_mask keeps the debug N x N score path; auto uses neighbor tables "
            "for sparse methods; auto_split uses local/cross split for sparse methods; "
            "auto_blockpair groups cross edges by block pair."
        ),
    )
    parser.add_argument("--num-keys", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-values", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-or-remote", choices=["local", "remote", "unknown"])
    parser.add_argument("--log-path")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()
