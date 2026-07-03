from __future__ import annotations

from src.tasks.common import JsonlDataset, TaskSpec, load_split


def train_dataset(spec: TaskSpec) -> JsonlDataset:
    return load_split(spec, spec.train_split)


def eval_dataset(spec: TaskSpec) -> JsonlDataset:
    return load_split(spec, spec.eval_split)

