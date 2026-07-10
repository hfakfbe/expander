from __future__ import annotations

import math
from typing import Any

from src.config.schema import (
    LOCAL_MODES,
    METHODS,
    NORM_TYPES,
    OPTIMIZERS,
    REQUIRED_FIELDS,
    SCHEDULERS,
    TASKS,
    get_path,
)


class ConfigError(ValueError):
    pass


def require_fields(config: dict[str, Any]) -> None:
    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        try:
            get_path(config, field)
        except KeyError:
            missing.append(field)
    if missing:
        raise ConfigError(f"config missing required fields: {', '.join(missing)}")


def validate_config(config: dict[str, Any]) -> None:
    require_fields(config)
    task = str(config["task"]["name"])
    if task not in TASKS:
        raise ConfigError(f"unknown task: {task}")
    method = str(config["attention"]["method"])
    if method not in METHODS:
        raise ConfigError(f"unknown method: {method}")
    if str(config["attention"]["local_mode"]) not in LOCAL_MODES:
        raise ConfigError("local_mode must be sliding_window")
    if str(config["training"]["scheduler"]) not in SCHEDULERS:
        raise ConfigError("scheduler must be constant or cosine")
    if str(config["training"]["optimizer"]) not in OPTIMIZERS:
        raise ConfigError("optimizer must be adamw")
    if int(config["training"]["max_steps"]) <= 0:
        raise ConfigError("training.max_steps must be positive")
    epochs = config["training"].get("epochs")
    if epochs is not None:
        epoch_value = float(epochs)
        if not math.isfinite(epoch_value) or epoch_value <= 0.0:
            raise ConfigError("training.epochs must be positive or null")
    if int(config["model"]["dim"]) % int(config["model"]["num_heads"]) != 0:
        raise ConfigError("model.dim must be divisible by model.num_heads")
    if str(config["model"]["norm_type"]) not in NORM_TYPES:
        raise ConfigError("model.norm_type must be layernorm, rmsnorm, or none")
    if int(config["training"]["batch_size"]) != int(config["training"]["minibatch_size"]) * int(
        config["training"]["gradient_accumulation_steps"]
    ):
        raise ConfigError("batch_size must equal minibatch_size * gradient_accumulation_steps")
    local_window_size = int(config["attention"]["local_window_size"])
    if local_window_size <= 0:
        raise ConfigError("local_window_size must be positive")
    if local_window_size > int(config["task"]["sequence_length"]):
        raise ConfigError("local_window_size must not exceed task.sequence_length")
    if bool(config["attention"]["include_local_edges"]) and local_window_size < int(config["attention"]["B"]):
        raise ConfigError("local_window_size must be at least B when local edges are enabled")
    if method == "random_regular":
        degree = config["attention"].get("random_regular", {}).get("degree")
        density = config["attention"].get("random_regular", {}).get("density", config["attention"].get("density"))
        if degree is None and density is None:
            raise ConfigError("random_regular requires degree or density")
    if method == "zigzag_logm" and not bool(config["attention"]["zigzag_logm"]["use_multiplicity_logm"]):
        raise ConfigError("zigzag_logm must use multiplicity/log-m")
    if method == "zigzag_boolean" and bool(config["attention"]["zigzag_boolean"].get("use_multiplicity_logm")):
        raise ConfigError("zigzag_boolean must not use multiplicity/log-m")
    if method in {"zigzag_logm", "zigzag_boolean"}:
        if int(config["attention"]["q"]) * int(config["attention"]["B"]) != int(config["task"]["sequence_length"]):
            raise ConfigError("zigzag q * B must equal task.sequence_length")
    seeds = config["attention"]["per_layer_graph_seeds"]
    if seeds is not None and len(seeds) != int(config["model"]["num_layers"]):
        raise ConfigError("per_layer_graph_seeds length must equal num_layers")
    memory = config["memory_rollout"]
    if not isinstance(memory["enabled"], bool):
        raise ConfigError("memory_rollout.enabled must be boolean")
    alpha = float(memory["alpha"])
    if alpha < 0.0 or alpha > 1.0:
        raise ConfigError("memory_rollout.alpha must be in [0, 1]")
    if not math.isfinite(float(memory["injection_scale"])):
        raise ConfigError("memory_rollout.injection_scale must be finite")
    if str(memory["head_merge"]) != "mean":
        raise ConfigError("memory_rollout.head_merge must be mean")
    if str(memory["update"]) != "lazy":
        raise ConfigError("memory_rollout.update must be lazy")
    if str(memory["initial_state"]) != "input":
        raise ConfigError("memory_rollout.initial_state must be input")
    if int(config["task"]["sequence_length"]) <= 0:
        raise ConfigError("task.sequence_length must be positive")
    if int(config["task"]["vocab_size"]) <= 0 or int(config["task"]["output_size"]) <= 0:
        raise ConfigError("task vocab_size/output_size must be positive")


def validate_task_matches_config(task_name: str, config: dict[str, Any]) -> None:
    configured = str(config["task"]["name"])
    if task_name != configured:
        raise ConfigError(f"config task mismatch: config has {configured}, requested {task_name}")
