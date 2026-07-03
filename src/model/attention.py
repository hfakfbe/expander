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
        self.memory_routes = graph.memory_routes

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if self.log_m is not None:
            scores = scores + self.log_m[None, None, :, :].to(dtype=scores.dtype, device=scores.device)
        scores = scores.masked_fill(~self.mask[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(weights, v)


def apply_memory_routes(hidden: torch.Tensor, routes, *, mode: str | None = None, scale: float | None = None) -> torch.Tensor:
    if routes is None or not routes.src:
        return hidden
    src = torch.tensor(routes.src, dtype=torch.long, device=hidden.device)
    dst = torch.tensor(routes.dst, dtype=torch.long, device=hidden.device)
    route_mode = mode or routes.mode
    route_scale = routes.scale if scale is None else float(scale)
    out = hidden.clone()
    carried = hidden.index_select(1, src) * float(route_scale)
    if route_mode == "memory_replace":
        out[:, dst, :] = carried
    elif route_mode == "memory_residual":
        out[:, dst, :] = out[:, dst, :] + carried
    else:
        raise ValueError(f"unknown memory route mode: {route_mode}")
    return out
