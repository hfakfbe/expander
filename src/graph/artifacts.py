from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.graph.diagnostics import graph_diagnostics, per_layer_identity
from src.graph.structures import LayerGraph
from src.io.hashing import file_sha256, sha256_json
from src.io.json import read_json, write_json


def layer_graph_to_artifact(graph: LayerGraph, diagnostics: dict) -> dict[str, Any]:
    edges: list[list[int]] = []
    for src in range(graph.seq_len):
        dst = torch.nonzero(graph.mask[src], as_tuple=False).flatten().detach().cpu().tolist()
        edges.extend([[src, int(item)] for item in dst])
    payload: dict[str, Any] = {
        "schema": "graph_artifact",
        "layer_index": graph.layer_index,
        "method": graph.method,
        "seq_len": graph.seq_len,
        "seed": graph.seed,
        "edges": edges,
        "diagnostics": diagnostics,
    }
    if graph.counts is not None:
        payload["multiplicity"] = [
            [src, int(dst), int(value)]
            for src, counts in enumerate(graph.counts)
            for dst, value in counts.items()
        ]
    payload["artifact_id"] = "graph_" + sha256_json(payload)[:16]
    return payload


def write_graph_artifacts(
    graphs: list[LayerGraph],
    root: Path,
    *,
    local_window_size: int,
    causal: bool,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    identities = per_layer_identity(graphs)
    records: list[dict[str, Any]] = []
    for graph, identity in zip(graphs, identities):
        diag = graph_diagnostics(graph, local_window_size, causal)
        diag["per_layer_graph_identity"] = identity
        payload = layer_graph_to_artifact(graph, diag)
        path = root / f"layer_{graph.layer_index:03d}.json"
        write_json(path, payload)
        records.append(
            {
                "layer_index": graph.layer_index,
                "path": str(path),
                "sha256": file_sha256(path),
                "artifact_id": payload["artifact_id"],
                "diagnostics": diag,
            }
        )
    manifest = {
        "schema": "graph_artifact_manifest",
        "layers": records,
        "per_layer_graph_identity": identities,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    records.append({"layer_index": "manifest", "path": str(manifest_path), "sha256": file_sha256(manifest_path)})
    return records


def load_graph_artifact(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "graph_artifact":
        raise ValueError(f"not a graph artifact: {path}")
    return payload


def require_graph_artifact_policy(config: dict[str, Any]) -> None:
    policy = str(config["attention"]["graph_artifact_policy"])
    method = str(config["attention"]["method"])
    root = Path(config["attention"]["graph_artifact_root"])
    if policy == "reuse" and method not in {"dense", "local"} and not (root / "manifest.json").exists():
        raise FileNotFoundError(f"method {method} requires existing graph artifact manifest at {root}")
    if policy not in {"reuse", "regenerate"}:
        raise ValueError("graph_artifact_policy must be reuse or regenerate")
