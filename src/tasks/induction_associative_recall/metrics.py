from __future__ import annotations

import torch

from src.tasks.common import TaskBatch, sequence_loss_and_metrics


def loss_and_metrics(token_logits: torch.Tensor, class_logits: torch.Tensor | None, batch: TaskBatch, spec) -> tuple[torch.Tensor, dict]:
    loss, metrics = sequence_loss_and_metrics(token_logits, batch, "induction_associative_recall")
    metrics["position_value_token_accuracy"] = metrics["token_accuracy"]
    return loss, metrics

