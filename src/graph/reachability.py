from __future__ import annotations

from src.graph.structures import LayerGraph


def reachable_within_layers(graphs: list[LayerGraph], source: int) -> set[int]:
    reached = {int(source)}
    for graph in graphs:
        new = set(reached)
        for src in list(reached):
            dsts = graph.mask[src].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            new.update(int(dst) for dst in dsts)
        reached = new
    return reached

