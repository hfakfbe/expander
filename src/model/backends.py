from __future__ import annotations

import torch
import torch.nn as nn

from src.graph.structures import LayerGraph
from src.model.attention import AttentionBackend


class BackendBundle(nn.Module):
    def __init__(self, graphs: list[LayerGraph], attention_dropout: float):
        super().__init__()
        self.backends = nn.ModuleList([AttentionBackend(graph, attention_dropout) for graph in graphs])

    def __len__(self) -> int:
        return len(self.backends)

    def __getitem__(self, index: int) -> AttentionBackend:
        return self.backends[index]


def build_backend_bundle(graphs: list[LayerGraph], attention_dropout: float) -> BackendBundle:
    return BackendBundle(graphs, attention_dropout)

