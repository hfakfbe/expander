from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, TaskSpec, make_int_sequence_batch


def make_batch(rows: list[dict], spec: TaskSpec, device: torch.device) -> TaskBatch:
    tokens, pad_mask, token_count = make_int_sequence_batch(rows, spec, device)
    if spec.source_length is None or spec.marker_token_id is None:
        raise ValueError("copy requires source_length and marker_token_id")
    target_len = int(spec.target_length)
    start = int(spec.source_length)
    target_positions = torch.arange(start, start + target_len, dtype=torch.long).repeat(len(rows), 1)
    targets = torch.zeros((len(rows), target_len), dtype=torch.long)
    target_mask = torch.ones((len(rows), target_len), dtype=torch.bool)
    for index, row in enumerate(rows):
        row_target = [int(item) for item in row["target"]]
        row_input = [int(item) for item in row["input"]]
        if row_input[:target_len] != row_target:
            raise ValueError("copy target must equal source prefix")
        if row_input[start : start + target_len] != [spec.marker_token_id] * target_len:
            raise ValueError("copy marker/readout suffix is invalid")
        targets[index] = torch.tensor(row_target, dtype=torch.long)
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

