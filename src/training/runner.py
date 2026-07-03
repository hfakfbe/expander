from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

import torch

from src.config.loading import write_resolved_config
from src.graph.artifacts import write_graph_artifacts
from src.graph.generation import build_layer_graphs
from src.io.manifest import build_run_manifest, write_run_manifest
from src.model.backends import build_backend_bundle
from src.model.transformer import SequenceTransformer, model_config_from_resolved
from src.tasks.common import JsonlDataset, TaskSpec, load_split, split_path, task_spec_from_config
from src.tasks.registry import get_batcher, get_loss
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.evaluation import evaluate
from src.training.results import append_metric, write_final_metrics
from src.training.schedule import apply_learning_rate, learning_rate


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_dtype(config: dict[str, Any]) -> torch.dtype:
    name = str(config["training"]["dtype"])
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float64", "fp64"}:
        return torch.float64
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unknown dtype: {name}")


def _command_string(argv: list[str] | None) -> str:
    if argv is None:
        import sys

        argv = sys.argv
    return " ".join(shlex.quote(item) for item in argv)


def _step_rows(dataset: JsonlDataset, config: dict[str, Any], micro_step: int) -> list[dict]:
    batch_size = int(config["training"]["minibatch_size"])
    seed = int(config["training"]["seed"])
    buffer_size = int(config["training"].get("shuffle_buffer_size", 4096))
    needed_start = int(micro_step) * batch_size
    rows: list[dict] = []
    seen = 0
    for batch in dataset.batches(batch_size, shuffle=True, seed=seed, buffer_size=buffer_size):
        if seen >= needed_start:
            rows = batch
            break
        seen += len(batch)
    if not rows:
        iterator = dataset.batches(batch_size, shuffle=True, seed=seed + micro_step + 1, buffer_size=buffer_size)
        rows = next(iterator)
    return rows


def build_runtime(config: dict[str, Any], device: torch.device):
    graphs = build_layer_graphs(config, device)
    backends = build_backend_bundle(graphs, float(config["model"]["attention_dropout"])).to(device)
    dtype = select_dtype(config)
    model = SequenceTransformer(model_config_from_resolved(config)).to(device=device, dtype=dtype)
    return model, backends, graphs


def run_check(config: dict[str, Any], output_dir: Path, argv: list[str] | None = None) -> dict:
    device = select_device(str(config["training"]["device"]))
    spec = task_spec_from_config(config)
    train_data = load_split(spec, spec.train_split)
    model, backends, graphs = build_runtime(config, device)
    batcher = get_batcher(spec.name)
    rows = next(train_data.batches(int(config["training"]["minibatch_size"]), shuffle=False, seed=0))
    batch = batcher(rows, spec, device)
    token_logits, class_logits = model(batch.tokens, batch.pad_mask, backends)
    loss, metrics = get_loss(spec.name)(token_logits, class_logits, batch, spec)
    graph_records = []
    if bool(config["run"]["save_graph_artifacts"]):
        graph_records = write_graph_artifacts(
            graphs,
            Path(config["attention"]["graph_artifact_root"]),
            local_window_size=int(config["attention"]["local_window_size"]),
            causal=bool(config["attention"]["causal"]),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_dir)
    final = {"mode": "check", "loss": float(loss.item()), **metrics}
    write_final_metrics(output_dir / "final_metrics.json", final)
    manifest = build_run_manifest(
        config=config,
        command=_command_string(argv),
        output_dir=output_dir,
        dataset_paths={spec.train_split: split_path(spec, spec.train_split)},
        graph_artifacts=graph_records,
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return final


def run_final_eval(config: dict[str, Any], output_dir: Path, checkpoint: Path | None = None, argv: list[str] | None = None) -> dict:
    device = select_device(str(config["training"]["device"]))
    spec = task_spec_from_config(config)
    eval_data = load_split(spec, spec.eval_split)
    model, backends, graphs = build_runtime(config, device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model=model, map_location=device)
    metrics = evaluate(
        model=model,
        backends=backends,
        dataset=eval_data,
        spec=spec,
        batch_size=int(config["training"]["minibatch_size"]),
        device=device,
        max_batches=int(config["training"]["final_eval_batches"]),
    )
    graph_records = []
    if bool(config["run"]["save_graph_artifacts"]):
        graph_records = write_graph_artifacts(
            graphs,
            Path(config["attention"]["graph_artifact_root"]),
            local_window_size=int(config["attention"]["local_window_size"]),
            causal=bool(config["attention"]["causal"]),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_dir)
    final = {"mode": "final_eval", **metrics}
    write_final_metrics(output_dir / "final_metrics.json", final)
    manifest = build_run_manifest(
        config=config,
        command=_command_string(argv),
        output_dir=output_dir,
        dataset_paths={spec.eval_split: split_path(spec, spec.eval_split)},
        graph_artifacts=graph_records,
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return final


def train(config: dict[str, Any], output_dir: Path, argv: list[str] | None = None) -> dict:
    torch.manual_seed(int(config["training"]["seed"]))
    device = select_device(str(config["training"]["device"]))
    spec = task_spec_from_config(config)
    train_data = load_split(spec, spec.train_split)
    eval_data = load_split(spec, spec.eval_split)
    model, backends, graphs = build_runtime(config, device)
    if str(config["training"]["optimizer"]) != "adamw":
        raise ValueError("only adamw optimizer is supported")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start_step = 0
    resume_path = config["training"].get("resume_checkpoint") or config["run"].get("resume_checkpoint")
    if resume_path:
        start_step = load_checkpoint(Path(resume_path), model=model, optimizer=optimizer, map_location=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    if start_step == 0 and metrics_path.exists():
        metrics_path.unlink()
    graph_records = []
    if bool(config["run"]["save_graph_artifacts"]):
        graph_records = write_graph_artifacts(
            graphs,
            Path(config["attention"]["graph_artifact_root"]),
            local_window_size=int(config["attention"]["local_window_size"]),
            causal=bool(config["attention"]["causal"]),
        )
    loss_fn = get_loss(spec.name)
    batcher = get_batcher(spec.name)
    started = time.perf_counter()
    max_steps = int(config["training"]["max_steps"])
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    last_metrics: dict[str, Any] = {}
    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        token_count = 0
        example_count = 0
        for accum_index in range(grad_accum):
            micro_step = (step - 1) * grad_accum + accum_index
            rows = _step_rows(train_data, config, micro_step)
            batch = batcher(rows, spec, device)
            token_logits, class_logits = model(batch.tokens, batch.pad_mask, backends)
            loss, metrics = loss_fn(token_logits, class_logits, batch, spec)
            (loss / grad_accum).backward()
            step_loss += float(loss.item())
            token_count += int(metrics.get("tokens", batch.token_count))
            example_count += int(metrics.get("examples", batch.example_count))
            last_metrics = metrics
        grad_clip = float(config["training"]["grad_clip_norm"])
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = learning_rate(config, step)
        apply_learning_rate(optimizer, lr)
        optimizer.step()
        should_eval = step == 1 or step % int(config["training"]["eval_interval"]) == 0 or step == max_steps
        eval_metrics = {}
        if should_eval:
            eval_metrics = evaluate(
                model=model,
                backends=backends,
                dataset=eval_data,
                spec=spec,
                batch_size=int(config["training"]["minibatch_size"]),
                device=device,
                max_batches=int(config["training"]["eval_batches"]),
            )
        if step == 1 or step % int(config["training"]["log_step"]) == 0 or step == max_steps:
            row = {
                "step": step,
                "method": config["attention"]["method"],
                "task": spec.name,
                "learning_rate": lr,
                "train_loss": step_loss / grad_accum,
                "examples": example_count,
                "tokens": token_count,
                "elapsed_sec": time.perf_counter() - started,
                **{f"train_{key}": value for key, value in last_metrics.items() if isinstance(value, (int, float))},
                **{f"eval_{key}": value for key, value in eval_metrics.items() if isinstance(value, (int, float))},
            }
            append_metric(metrics_path, row)
        if bool(config["run"]["save_checkpoints"]) and int(config["training"]["checkpoint_interval"]) > 0:
            if step % int(config["training"]["checkpoint_interval"]) == 0 or step == max_steps:
                save_checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt", model=model, optimizer=optimizer, step=step, config=config)
    final_metrics = evaluate(
        model=model,
        backends=backends,
        dataset=eval_data,
        spec=spec,
        batch_size=int(config["training"]["minibatch_size"]),
        device=device,
        max_batches=int(config["training"]["final_eval_batches"]),
    )
    final = {
        "mode": "train",
        "status": "ok",
        "final_step": max_steps,
        "train_elapsed_sec": time.perf_counter() - started,
        **final_metrics,
    }
    write_final_metrics(output_dir / "final_metrics.json", final)
    manifest = build_run_manifest(
        config=config,
        command=_command_string(argv),
        output_dir=output_dir,
        dataset_paths={
            spec.train_split: split_path(spec, spec.train_split),
            spec.eval_split: split_path(spec, spec.eval_split),
        },
        graph_artifacts=graph_records,
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return final


def run(config: dict[str, Any], mode: str, argv: list[str] | None = None) -> dict:
    output_dir = Path(config["run"]["output_root"]) / str(config["run"]["run_id"])
    if mode == "check":
        return run_check(config, output_dir, argv)
    if mode == "train":
        return train(config, output_dir, argv)
    if mode in {"final-eval", "final_eval"}:
        checkpoint = config["training"].get("resume_checkpoint") or config["run"].get("resume_checkpoint")
        return run_final_eval(config, output_dir, Path(checkpoint) if checkpoint else None, argv)
    raise ValueError(f"unknown mode: {mode}")
