from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_task(config_path: Path | None) -> str:
    if config_path is None:
        return "copy"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config.get("task", {}).get("name", "copy"))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args()
    task = parse_task(known.config)
    if task == "copy":
        from synthetic_mvp_core.runner import main as run_copy

        run_copy()
        return
    if task == "wikitext":
        from wikitext2_eval import main as run_wikitext

        run_wikitext()
        return
    raise SystemExit(f"scripts/run_experiment.py does not handle task={task!r}")


if __name__ == "__main__":
    main()
