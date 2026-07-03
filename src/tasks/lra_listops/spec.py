from __future__ import annotations

from pathlib import Path

from src.io.json import read_json
from src.tasks.common import TaskSpec, task_spec_from_config

TASK_NAME = "lra_listops"


def load_task_spec(path: Path) -> dict:
    return read_json(path)


def validate_task_spec(spec: TaskSpec) -> None:
    if spec.name != TASK_NAME:
        raise ValueError("lra_listops task spec expected")
    if spec.loss_type != "classification":
        raise ValueError("lra_listops must be classification")


def from_config(config: dict) -> TaskSpec:
    spec = task_spec_from_config(config)
    validate_task_spec(spec)
    return spec

