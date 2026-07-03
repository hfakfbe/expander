from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, classification_loss_and_metrics


def loss_and_metrics(token_logits: torch.Tensor, class_logits: torch.Tensor | None, batch: TaskBatch, spec) -> tuple[torch.Tensor, dict]:
    if class_logits is None:
        raise ValueError("lra_listops requires class logits")
    return classification_loss_and_metrics(class_logits, batch, "listops")

