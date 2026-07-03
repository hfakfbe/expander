from __future__ import annotations

from pathlib import Path

from src.io.json import read_json
from src.tasks.common import TaskSpec, task_spec_from_config

TASK_NAME = "selective_copy"


def load_task_spec(path: Path) -> dict:
    return read_json(path)


def validate_task_spec(spec: TaskSpec) -> None:
    if spec.name != TASK_NAME:
        raise ValueError("selective_copy task spec expected")
    if spec.target_length <= 0:
        raise ValueError("selective_copy requires target_length")


def from_config(config: dict) -> TaskSpec:
    spec = task_spec_from_config(config)
    validate_task_spec(spec)
    return spec

