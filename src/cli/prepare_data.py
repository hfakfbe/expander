from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config.loading import add_common_config_args, resolve_config, write_resolved_config
from src.graph.artifacts import write_graph_artifacts
from src.graph.generation import build_layer_graphs


def prepare(config: dict, output: Path | None = None) -> dict:
    output_dir = output or Path(config["run"]["output_root"]) / str(config["run"]["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = write_resolved_config(config, output_dir)
    graph_records = []
    if bool(config["run"]["save_graph_artifacts"]):
        graphs = build_layer_graphs(config, torch.device("cpu"))
        graph_records = write_graph_artifacts(
            graphs,
            Path(config["attention"]["graph_artifact_root"]),
            local_window_size=int(config["attention"]["local_window_size"]),
            causal=bool(config["attention"]["causal"]),
        )
    return {
        "status": "ok",
        "task": config["task"]["name"],
        "method": config["attention"]["method"],
        "resolved_config_path": str(resolved_path),
        "graph_artifacts": graph_records,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare resolved config and graph artifacts.")
    add_common_config_args(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = resolve_config(args.config, requested_task=args.task, overrides=args.override)
    result = prepare(config, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

