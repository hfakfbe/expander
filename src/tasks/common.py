from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_root: Path
    train_split: str
    eval_split: str
    sequence_length: int
    target_length: int
    vocab_size: int
    output_size: int
    loss_type: str
    pad_token_id: int = 0
    marker_token_id: int | None = None
    source_length: int | None = None


@dataclass
class TaskBatch:
    tokens: torch.Tensor
    pad_mask: torch.Tensor
    target_positions: torch.Tensor | None
    targets: torch.Tensor | None
    target_mask: torch.Tensor | None
    class_targets: torch.Tensor | None
    example_count: int
    token_count: int


class JsonlDataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def rows(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)

    def count(self) -> int:
        total = 0
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    total += 1
        return total

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        buffer_size: int = 4096,
        limit: int | None = None,
    ) -> Iterator[list[dict]]:
        iterator = self._shuffled_rows(seed, buffer_size, limit) if shuffle else self._limited_rows(limit)
        batch: list[dict] = []
        for row in iterator:
            batch.append(row)
            if len(batch) == int(batch_size):
                yield batch
                batch = []
        if batch:
            yield batch

    def _limited_rows(self, limit: int | None) -> Iterator[dict]:
        for index, row in enumerate(self.rows()):
            if limit is not None and index >= int(limit):
                break
            yield row

    def _shuffled_rows(self, seed: int, buffer_size: int, limit: int | None) -> Iterator[dict]:
        rng = random.Random(int(seed))
        source = self._limited_rows(limit)
        buffer: list[dict] = []
        for _ in range(max(1, int(buffer_size))):
            try:
                buffer.append(next(source))
            except StopIteration:
                break
        while buffer:
            index = rng.randrange(len(buffer))
            row = buffer[index]
            try:
                buffer[index] = next(source)
            except StopIteration:
                buffer.pop(index)
            yield row


def split_path(spec: TaskSpec, split: str) -> Path:
    return spec.dataset_root / f"{split}.jsonl"


def make_int_sequence_batch(rows: list[dict], spec: TaskSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokens = torch.full((len(rows), spec.sequence_length), spec.pad_token_id, dtype=torch.long)
    pad_mask = torch.zeros((len(rows), spec.sequence_length), dtype=torch.bool)
    token_count = 0
    for index, row in enumerate(rows):
        values = [int(item) for item in row["input"]]
        if len(values) > spec.sequence_length:
            raise ValueError(f"{spec.name} input length {len(values)} exceeds sequence_length={spec.sequence_length}")
        if values:
            tokens[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            pad_mask[index, : len(values)] = True
            token_count += len(values)
    return tokens.to(device), pad_mask.to(device), token_count


def gather_position_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = positions[:, :, None].expand(-1, -1, logits.shape[-1])
    return logits.gather(1, index)


def sequence_loss_and_metrics(logits: torch.Tensor, batch: TaskBatch, prefix: str) -> tuple[torch.Tensor, dict]:
    if batch.target_positions is None or batch.targets is None or batch.target_mask is None:
        raise ValueError("sequence metrics require target positions, targets, and mask")
    selected = gather_position_logits(logits, batch.target_positions)
    flat_mask = batch.target_mask.reshape(-1)
    flat_logits = selected.reshape(-1, selected.shape[-1])[flat_mask]
    flat_targets = batch.targets.reshape(-1)[flat_mask]
    loss = F.cross_entropy(flat_logits, flat_targets)
    pred = selected.argmax(dim=-1)
    correct = (pred == batch.targets) & batch.target_mask
    token_total = int(batch.target_mask.sum().item())
    token_accuracy = float(correct.sum().item() / max(token_total, 1))
    sequence_accuracy = float((((pred == batch.targets) | ~batch.target_mask).all(dim=1)).float().mean().item())
    return loss, {
        "loss": float(loss.detach().item()),
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        f"{prefix}_token_accuracy": token_accuracy,
        f"{prefix}_sequence_accuracy": sequence_accuracy,
        "tokens": token_total,
        "examples": batch.example_count,
    }


def classification_loss_and_metrics(logits: torch.Tensor, batch: TaskBatch, prefix: str) -> tuple[torch.Tensor, dict]:
    if batch.class_targets is None:
        raise ValueError("classification metrics require class_targets")
    loss = F.cross_entropy(logits, batch.class_targets)
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == batch.class_targets).float().mean().item())
    return loss, {
        "loss": float(loss.detach().item()),
        "accuracy": accuracy,
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_macro_accuracy": accuracy,
        "tokens": int(batch.class_targets.numel()),
        "examples": batch.example_count,
    }


def task_spec_from_config(config: dict) -> TaskSpec:
    task = config["task"]
    return TaskSpec(
        name=str(task["name"]),
        dataset_root=Path(task["dataset_root"]),
        train_split=str(task["train_split"]),
        eval_split=str(task["eval_split"]),
        sequence_length=int(task["sequence_length"]),
        target_length=int(task.get("target_length", 1)),
        vocab_size=int(task["vocab_size"]),
        output_size=int(task["output_size"]),
        loss_type=str(task["loss_type"]),
        pad_token_id=int(task.get("pad_token_id", 0)),
        marker_token_id=None if task.get("marker_token_id") is None else int(task["marker_token_id"]),
        source_length=None if task.get("source_length") is None else int(task["source_length"]),
    )


def load_split(spec: TaskSpec, split: str) -> JsonlDataset:
    return JsonlDataset(split_path(spec, split))


TaskBatcher = Callable[[list[dict], TaskSpec, torch.device], TaskBatch]

