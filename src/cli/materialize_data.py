from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config.loading import add_common_config_args, resolve_config
from src.io.hashing import file_sha256
from src.io.json import write_json


def materialize(config: dict) -> dict:
    task = config["task"]
    root = Path(task["dataset_root"])
    root.mkdir(parents=True, exist_ok=True)
    materialize_cfg = config.get("materialize", {})
    source_dir = materialize_cfg.get("source_dir")
    if source_dir:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(source)
        for split in [task["train_split"], task["eval_split"]]:
            src = source / f"{split}.jsonl"
            dst = root / f"{split}.jsonl"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copyfile(src, dst)
    required = [root / f"{task['train_split']}.jsonl", root / f"{task['eval_split']}.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing materialized splits: {missing}")
    checksums = {path.name: file_sha256(path) for path in required}
    write_json(root / "checksums.generated.json", checksums)
    return {"status": "ok", "task": task["name"], "dataset_root": str(root), "checksums": checksums}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize task data from configured sources.")
    add_common_config_args(parser)
    args = parser.parse_args(argv)
    config = resolve_config(args.config, requested_task=args.task, overrides=args.override)
    print(json.dumps(materialize(config), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

