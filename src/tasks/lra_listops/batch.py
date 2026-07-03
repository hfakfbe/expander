from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, TaskSpec
from src.tasks.lra_listops.encoder import encode_input


def make_batch(rows: list[dict], spec: TaskSpec, device: torch.device) -> TaskBatch:
    tokens = torch.full((len(rows), spec.sequence_length), spec.pad_token_id, dtype=torch.long)
    pad_mask = torch.zeros((len(rows), spec.sequence_length), dtype=torch.bool)
    class_targets = torch.zeros((len(rows),), dtype=torch.long)
    token_count = 0
    for index, row in enumerate(rows):
        values = encode_input(row["input"])
        if len(values) > spec.sequence_length:
            raise ValueError(f"lra_listops input length {len(values)} exceeds sequence_length={spec.sequence_length}")
        if values:
            tokens[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            pad_mask[index, : len(values)] = True
            token_count += len(values)
        class_targets[index] = int(row["target"])
    return TaskBatch(
        tokens=tokens.to(device),
        pad_mask=pad_mask.to(device),
        target_positions=None,
        targets=None,
        target_mask=None,
        class_targets=class_targets.to(device),
        example_count=len(rows),
        token_count=token_count,
    )

