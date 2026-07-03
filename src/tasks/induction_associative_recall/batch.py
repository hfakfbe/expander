from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, TaskSpec, make_int_sequence_batch


def make_batch(rows: list[dict], spec: TaskSpec, device: torch.device) -> TaskBatch:
    tokens, pad_mask, token_count = make_int_sequence_batch(rows, spec, device)
    max_targets = int(spec.target_length)
    target_positions = torch.zeros((len(rows), max_targets), dtype=torch.long)
    targets = torch.zeros((len(rows), max_targets), dtype=torch.long)
    target_mask = torch.zeros((len(rows), max_targets), dtype=torch.bool)
    for index, row in enumerate(rows):
        entries = list(row["target"])
        if len(entries) > max_targets:
            raise ValueError("induction_associative_recall target length exceeds spec")
        input_len = len(row["input"])
        for offset, item in enumerate(entries):
            pos = int(item["position"])
            if not 0 <= pos < input_len:
                raise ValueError("induction_associative_recall target position outside input")
            target_positions[index, offset] = pos
            targets[index, offset] = int(item["value"])
            target_mask[index, offset] = True
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

