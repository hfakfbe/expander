from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config.loading import add_common_config_args, resolve_config
from src.io.hashing import file_sha256
from src.tasks.common import load_split, split_path, task_spec_from_config
from src.tasks.registry import get_batcher


def audit(config: dict) -> dict:
    spec = task_spec_from_config(config)
    batcher = get_batcher(spec.name)
    result = {"task": spec.name, "splits": {}, "batch_contract": {}}
    for split in [spec.train_split, spec.eval_split]:
        dataset = load_split(spec, split)
        path = split_path(spec, split)
        result["splits"][split] = {
            "path": str(path),
            "rows": dataset.count(),
            "sha256": file_sha256(path),
        }
    rows = next(load_split(spec, spec.train_split).batches(2, shuffle=False, seed=0))
    batch = batcher(rows, spec, torch.device("cpu"))
    result["batch_contract"] = {
        "tokens_shape": list(batch.tokens.shape),
        "has_sequence_targets": batch.targets is not None,
        "has_class_targets": batch.class_targets is not None,
        "token_count": batch.token_count,
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit task data.")
    add_common_config_args(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = resolve_config(args.config, requested_task=args.task, overrides=args.override)
    result = audit(config)
    if args.output:
        from src.io.json import write_json

        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

