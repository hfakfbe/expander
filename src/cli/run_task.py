from __future__ import annotations

import argparse
import json
import sys

from src.config.loading import add_common_config_args, resolve_config
from src.training.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured task.")
    add_common_config_args(parser)
    parser.add_argument("--mode", choices=["check", "train", "final-eval"], default="check")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args.config, requested_task=args.task, overrides=args.override)
    result = run(config, args.mode, sys.argv if argv is None else ["run_task", *argv])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

