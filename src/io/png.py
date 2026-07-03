from __future__ import annotations

from pathlib import Path


def write_empty_png(path: Path) -> None:
    # Tiny valid 1x1 transparent PNG for optional curve outputs when plotting libs are unavailable.
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c636000000200015d0b2a00000000049454e44ae426082"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
