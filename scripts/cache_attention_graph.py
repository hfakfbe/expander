import argparse
import json
from pathlib import Path

import torch

from synthetic_mvp import (
    build_attention_mask,
    build_cross_mask,
    cross_neighbors_to_block_pair_index,
    mask_metrics,
    mask_to_neighbors,
)


def sorted_edge_index(mask: torch.Tensor) -> torch.Tensor:
    edge_index = torch.nonzero(mask, as_tuple=False).to(torch.long)
    if edge_index.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    order = edge_index[:, 0] * mask.shape[1] + edge_index[:, 1]
    return edge_index[torch.argsort(order)].cpu()


def block_pair_summary(records: torch.Tensor) -> list[dict]:
    if records.numel() == 0:
        return []
    summary = []
    unique_pairs, counts = torch.unique(records[:, :2], dim=0, return_counts=True)
    for pair, count in zip(unique_pairs.tolist(), counts.tolist(), strict=True):
        summary.append(
            {
                "source_block": int(pair[0]),
                "target_block": int(pair[1]),
                "edge_count": int(count),
            }
        )
    return summary


def tensor_shape(value: torch.Tensor | None) -> list[int] | None:
    return list(value.shape) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["local", "random", "zigzag"], required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.seq_len % args.block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")

    device = torch.device("cpu")
    mask = build_attention_mask(
        args.method, args.seq_len, args.block_size, args.degree, device, args.seed
    )
    cross_mask = build_cross_mask(
        args.method, args.seq_len, args.block_size, args.degree, device, args.seed
    )
    neighbors, valid_neighbors = mask_to_neighbors(mask)
    cross_neighbors, valid_cross_neighbors = mask_to_neighbors(cross_mask)
    edge_index = sorted_edge_index(mask)
    cross_edge_index = sorted_edge_index(cross_mask)
    block_pair_index = cross_neighbors_to_block_pair_index(
        cross_neighbors, valid_cross_neighbors, args.block_size
    ).cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(neighbors.cpu(), args.output_dir / "neighbors.pt")
    torch.save(valid_neighbors.cpu(), args.output_dir / "neighbors_valid.pt")
    torch.save(cross_neighbors.cpu(), args.output_dir / "cross_neighbors.pt")
    torch.save(valid_cross_neighbors.cpu(), args.output_dir / "cross_neighbors_valid.pt")
    torch.save(edge_index, args.output_dir / "edge_index.pt")
    torch.save(cross_edge_index, args.output_dir / "cross_edge_index.pt")
    torch.save(block_pair_index, args.output_dir / "block_pair_index.pt")

    metadata = {
        "method": args.method,
        "seq_len": args.seq_len,
        "block_size": args.block_size,
        "degree": args.degree,
        "seed": args.seed,
        "num_blocks": args.seq_len // args.block_size,
        "mask": mask_metrics(mask, args.method, args.block_size, args.degree),
        "cross_pair_count": int(cross_mask.sum().item()),
        "neighbors_shape": tensor_shape(neighbors),
        "cross_neighbors_shape": tensor_shape(cross_neighbors),
        "edge_index_shape": tensor_shape(edge_index),
        "cross_edge_index_shape": tensor_shape(cross_edge_index),
        "block_pair_index_shape": tensor_shape(block_pair_index),
        "block_pair_count": len(block_pair_summary(block_pair_index)),
        "block_pair_summary": block_pair_summary(block_pair_index),
        "block_pair_columns": [
            "source_block",
            "target_block",
            "source_port",
            "target_port",
            "source_token",
            "target_token",
            "cross_neighbor_slot",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
