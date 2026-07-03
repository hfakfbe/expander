from __future__ import annotations

from collections import Counter
from statistics import mean

import torch

from src.graph.structures import LayerGraph, local_edge_pairs


def _degree_stats(mask: torch.Tensor) -> dict:
    degrees = mask.sum(dim=1).detach().cpu().tolist()
    if not degrees:
        return {"degree_min": 0, "degree_mean": 0.0, "degree_max": 0}
    return {
        "degree_min": int(min(degrees)),
        "degree_mean": float(mean(degrees)),
        "degree_max": int(max(degrees)),
    }


def graph_diagnostics(graph: LayerGraph, local_window_size: int, causal: bool) -> dict:
    mask = graph.mask
    seq_len = int(mask.shape[0])
    total = seq_len * seq_len
    pair_count = int(mask.sum().item())
    local_edges = local_edge_pairs(seq_len, local_window_size, causal)
    local_count = sum(1 for src, dst in local_edges if bool(mask[src, dst].item()))
    remote_count = pair_count - local_count
    out = {
        "layer_index": graph.layer_index,
        "method": graph.method,
        "seed": graph.seed,
        "seq_len": seq_len,
        "actual_density": pair_count / max(total, 1),
        "attention_pair_count": pair_count,
        "local_edge_count": int(local_count),
        "remote_edge_count": int(remote_count),
        "has_log_m": graph.log_m is not None,
        "has_memory_routes": graph.memory_routes is not None,
        **_degree_stats(mask),
    }
    if graph.counts is not None:
        multiplicities = [value for counts in graph.counts for value in counts.values()]
        out.update(
            {
                "multiplicity_max": int(max(multiplicities)) if multiplicities else 0,
                "multiplicity_mean": float(mean(multiplicities)) if multiplicities else 0.0,
            }
        )
    else:
        out.update({"multiplicity_max": 1, "multiplicity_mean": 1.0})
    return out


def per_layer_identity(graphs: list[LayerGraph]) -> list[str]:
    identities: list[str] = []
    for graph in graphs:
        rows: list[tuple[int, ...]] = []
        for src in range(graph.seq_len):
            dst = torch.nonzero(graph.mask[src], as_tuple=False).flatten().detach().cpu().tolist()
            rows.append(tuple(int(v) for v in dst))
        identities.append(str(hash(tuple(rows))))
    return identities

