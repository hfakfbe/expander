from __future__ import annotations

import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.io.git import git_commit, git_dirty
from src.io.hashing import file_sha256, string_sha256
from src.io.json import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_manifest(
    *,
    config: dict[str, Any],
    command: str,
    output_dir: Path,
    dataset_paths: dict[str, Path],
    graph_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_sha256 = {
        split: file_sha256(path)
        for split, path in dataset_paths.items()
        if path.exists()
    }
    graph_sha256 = {
        str(item["path"]): item.get("sha256", "")
        for item in graph_artifacts
        if item.get("path")
    }
    return {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": command,
        "command_sha256": string_sha256(command),
        "output_dir": str(output_dir),
        "task": config["task"]["name"],
        "method": config["attention"]["method"],
        "final_config": config,
        "config_sha256": config.get("meta", {}).get("config_sha256", ""),
        "resolved_config_sha256": config.get("meta", {}).get("resolved_config_sha256", ""),
        "dataset_sha256": dataset_sha256,
        "graph_artifact_sha256": graph_sha256,
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)

