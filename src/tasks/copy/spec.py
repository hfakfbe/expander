from __future__ import annotations

from pathlib import Path

from src.io.json import read_json
from src.tasks.common import TaskSpec, task_spec_from_config

TASK_NAME = "copy"


def load_task_spec(path: Path) -> dict:
    return read_json(path)


def validate_task_spec(spec: TaskSpec) -> None:
    if spec.name != TASK_NAME:
        raise ValueError("copy task spec expected")
    if spec.marker_token_id is None or spec.source_length is None:
        raise ValueError("copy requires marker_token_id and source_length")


def from_config(config: dict) -> TaskSpec:
    spec = task_spec_from_config(config)
    validate_task_spec(spec)
    return spec

