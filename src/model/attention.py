from __future__ import annotations

import torch
import torch.nn as nn

from src.graph.structures import LayerGraph


class AttentionBackend(nn.Module):
    def __init__(self, graph: LayerGraph, dropout: float):
        super().__init__()
        self.kind = graph.method
        self.layer_index = graph.layer_index
        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer("mask", graph.mask, persistent=False)
        if graph.log_m is None:
            self.log_m = None
        else:
            self.register_buffer("log_m", graph.log_m, persistent=False)

    def transition(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if self.log_m is not None:
            scores = scores + self.log_m[None, None, :, :].to(dtype=scores.dtype, device=scores.device)
        scores = scores.masked_fill(~self.mask[None, None, :, :], torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        weights = self.dropout(self.transition(q, k))
        return torch.matmul(weights, v)
