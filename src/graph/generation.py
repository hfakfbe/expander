from __future__ import annotations

import math
from collections import Counter

import torch

from src.graph.artifacts import require_graph_artifact_policy
from src.graph.structures import (
    LayerGraph,
    counts_from_edges,
    counts_to_log_m,
    counts_to_mask,
    dense_mask,
    local_edge_pairs,
    merge_counts,
    random_regular_counts,
    sliding_window_mask,
    zigzag_counts,
)


def _layer_seed(config: dict, layer_index: int) -> int | None:
    seeds = config["attention"]["per_layer_graph_seeds"]
    if seeds is not None:
        return int(seeds[layer_index])
    base = int(config["attention"]["graph_seed"])
    if bool(config["attention"]["per_layer_random"]):
        return base + 1009 * (layer_index + 1)
    return base


def _base_local_counts(seq_len: int, config: dict, causal: bool) -> list[Counter[int]]:
    if not bool(config["attention"]["include_local_edges"]):
        return [Counter() for _ in range(seq_len)]
    return counts_from_edges(
        seq_len,
        local_edge_pairs(seq_len, int(config["attention"]["local_window_size"]), causal),
    )


def build_layer_graphs(config: dict, device: torch.device) -> list[LayerGraph]:
    require_graph_artifact_policy(config)
    method = str(config["attention"]["method"])
    seq_len = int(config["task"]["sequence_length"])
    layers = int(config["model"]["num_layers"])
    causal = bool(config["attention"]["causal"])
    local_window = int(config["attention"]["local_window_size"])
    graphs: list[LayerGraph] = []

    for layer_index in range(layers):
        seed = _layer_seed(config, layer_index)
        counts: list[Counter[int]] | None = None
        log_m = None
        if method == "dense":
            mask = dense_mask(seq_len, device, causal)
            seed = None
        elif method == "local":
            mask = sliding_window_mask(seq_len, local_window, device, causal)
            seed = None
        elif method == "random_regular":
            if seed is None:
                raise ValueError(f"{method} requires a graph seed")
            local_counts = _base_local_counts(seq_len, config, causal)
            exclude = local_edge_pairs(seq_len, local_window, causal) if bool(config["attention"]["include_local_edges"]) else set()
            method_cfg = config["attention"]["random_regular"]
            degree = method_cfg.get("degree")
            if degree is None:
                density = method_cfg.get("density", config["attention"]["density"])
                local_budget = local_window if bool(config["attention"]["include_local_edges"]) else 0
                degree = max(1, int(math.floor(float(density) * seq_len - local_budget)))
            remote = random_regular_counts(seq_len, int(degree), int(seed), exclude_edges=exclude)
            counts = merge_counts(local_counts, remote, boolean=True)
            mask = counts_to_mask(counts, seq_len, device, causal)
        elif method in {"zigzag_logm", "zigzag_boolean"}:
            if seed is None:
                raise ValueError(f"{method} requires a graph seed")
            local_counts = _base_local_counts(seq_len, config, causal)
            remote = zigzag_counts(seq_len, int(config["attention"]["B"]), int(config["attention"]["d"]), int(seed))
            counts = merge_counts(local_counts, remote, boolean=method == "zigzag_boolean")
            mask = counts_to_mask(counts, seq_len, device, causal)
            if method == "zigzag_logm":
                log_m = counts_to_log_m(counts, seq_len, device, causal)
        else:
            raise ValueError(f"unknown method: {method}")
        graphs.append(
            LayerGraph(
                layer_index=layer_index,
                method=method,
                seq_len=seq_len,
                mask=mask,
                counts=counts,
                log_m=log_m,
                seed=seed,
            )
        )
    return graphs
