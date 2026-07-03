from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config.schema import ALLOWED_CLI_OVERRIDES, deep_merge, default_config, set_path
from src.config.validation import ConfigError, validate_config, validate_task_matches_config
from src.io.hashing import file_sha256, sha256_json
from src.io.json import write_json


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be field=value, got {item!r}")
        field, raw = item.split("=", 1)
        if field not in ALLOWED_CLI_OVERRIDES:
            raise ConfigError(f"CLI override is not allowed by schema: {field}")
        set_path(out, field, parse_override(raw))
    return out


def resolve_config(path: Path, requested_task: str | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    raw = read_config(path)
    merged = deep_merge(default_config(), raw)
    if overrides:
        merged = apply_cli_overrides(merged, overrides)
    if requested_task is not None:
        validate_task_matches_config(requested_task, merged)
    validate_config(merged)
    merged.setdefault("meta", {})
    merged["meta"]["config_path"] = str(path)
    merged["meta"]["config_sha256"] = file_sha256(path)
    merged["meta"]["resolved_config_sha256"] = sha256_json(merged)
    return merged


def write_resolved_config(config: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "resolved_config.json"
    write_json(path, config)
    return path


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])

