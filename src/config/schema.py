from __future__ import annotations

from copy import deepcopy
from typing import Any


TASKS = {
    "copy",
    "selective_copy",
    "induction_associative_recall",
    "lra_listops",
}

METHODS = {
    "dense",
    "local",
    "random_regular",
    "zigzag_logm",
    "zigzag_boolean",
}

SCHEDULERS = {"constant", "cosine"}
OPTIMIZERS = {"adamw"}
LOCAL_MODES = {"sliding_window"}
NORM_TYPES = {"layernorm", "rmsnorm", "none"}

ALLOWED_CLI_OVERRIDES = {
    "task.name",
    "attention.method",
    "model.norm_type",
    "memory_rollout.enabled",
    "memory_rollout.alpha",
    "memory_rollout.injection_scale",
    "training.max_steps",
    "training.epochs",
    "training.seed",
    "training.device",
    "run.output_root",
    "run.run_id",
    "run.resume_checkpoint",
}

REQUIRED_FIELDS = (
    "task.name",
    "task.dataset_root",
    "task.train_split",
    "task.eval_split",
    "task.vocab_size",
    "task.output_size",
    "task.sequence_length",
    "model.num_layers",
    "model.dim",
    "model.dim_ffn",
    "model.num_heads",
    "model.activation",
    "model.dropout",
    "model.attention_dropout",
    "model.norm_type",
    "model.positional_encoding",
    "training.learning_rate",
    "training.warmup_steps",
    "training.scheduler",
    "training.min_learning_rate",
    "training.weight_decay",
    "training.optimizer",
    "training.max_steps",
    "training.batch_size",
    "training.minibatch_size",
    "training.gradient_accumulation_steps",
    "training.log_step",
    "training.eval_interval",
    "training.checkpoint_interval",
    "training.seed",
    "training.device",
    "training.dtype",
    "training.amp",
    "training.grad_clip_norm",
    "attention.method",
    "attention.causal",
    "attention.local_mode",
    "attention.local_window_size",
    "attention.include_local_edges",
    "attention.per_layer_random",
    "attention.graph_seed",
    "attention.per_layer_graph_seeds",
    "attention.q",
    "attention.B",
    "attention.d",
    "attention.density",
    "attention.graph_artifact_root",
    "attention.graph_artifact_policy",
    "memory_rollout.enabled",
    "memory_rollout.alpha",
    "memory_rollout.injection_scale",
    "memory_rollout.head_merge",
    "memory_rollout.update",
    "memory_rollout.initial_state",
    "run.output_root",
    "run.run_id",
    "run.save_checkpoints",
    "run.save_graph_artifacts",
    "run.save_metrics",
    "run.save_manifest",
)


SCHEMA_DEFAULTS: dict[str, Any] = {
    "model": {
        "rope": {
            "enabled": False,
            "theta": 10000.0,
        },
    },
    "training": {
        "epochs": None,
        "resume_checkpoint": None,
        "shuffle_buffer_size": 4096,
        "eval_batches": 1,
        "final_eval_batches": 1,
    },
    "attention": {
        "random_regular": {
            "degree": None,
            "density": None,
        },
        "zigzag_logm": {
            "use_multiplicity_logm": True,
        },
        "zigzag_boolean": {
            "use_multiplicity_logm": False,
        },
    },
    "memory_rollout": {
        "enabled": False,
        "alpha": 0.5,
        "injection_scale": 2.0,
        "head_merge": "mean",
        "update": "lazy",
        "initial_state": "input",
    },
    "run": {
        "manifest_path": None,
        "config_sha256_required": None,
        "dataset_sha256_required": None,
    },
}


def default_config() -> dict[str, Any]:
    return deepcopy(SCHEMA_DEFAULTS)


def get_path(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def set_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current: Any = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out
