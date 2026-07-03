from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, TaskSpec, make_int_sequence_batch


def make_batch(rows: list[dict], spec: TaskSpec, device: torch.device) -> TaskBatch:
    tokens, pad_mask, token_count = make_int_sequence_batch(rows, spec, device)
    target_len = int(spec.target_length)
    target_positions = torch.zeros((len(rows), target_len), dtype=torch.long)
    targets = torch.zeros((len(rows), target_len), dtype=torch.long)
    target_mask = torch.ones((len(rows), target_len), dtype=torch.bool)
    for index, row in enumerate(rows):
        values = [int(item) for item in row["target"]]
        if len(values) != target_len:
            raise ValueError("selective_copy target length mismatch")
        input_len = len(row["input"])
        start = input_len - target_len
        if start < 0:
            raise ValueError("selective_copy target does not fit input")
        target_positions[index] = torch.arange(start, input_len, dtype=torch.long)
        targets[index] = torch.tensor(values, dtype=torch.long)
    return TaskBatch(
        tokens=tokens,
        pad_mask=pad_mask,
        target_positions=target_positions.to(device),
        targets=targets.to(device),
        target_mask=target_mask.to(device),
        class_targets=None,
        example_count=len(rows),
        token_count=token_count,
    )

