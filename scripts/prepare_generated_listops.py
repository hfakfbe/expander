import argparse
import csv
import random
from pathlib import Path

import numpy as np


OPERATORS = ("MIN", "MAX", "MEDIAN", "SUM_MOD")
VALUES = tuple(range(10))
OPERATOR_P = 0.25


def generate_tree(depth: int, max_depth: int, max_args: int):
    if depth >= max_depth or random.random() > OPERATOR_P:
        return random.choice(VALUES)
    op = random.choice(OPERATORS)
    num_values = random.randint(2, max_args)
    return (op, [generate_tree(depth + 1, max_depth, max_args) for _ in range(num_values)])


def to_value(tree) -> int:
    if isinstance(tree, int):
        return tree
    op, children = tree
    values = [to_value(child) for child in children]
    if op == "MIN":
        return min(values)
    if op == "MAX":
        return max(values)
    if op == "MEDIAN":
        return int(np.median(values))
    if op == "SUM_MOD":
        return int(np.sum(values) % 10)
    raise ValueError(f"unknown op: {op}")


def to_tokens(tree) -> list[str]:
    if isinstance(tree, int):
        return [str(tree)]
    op, children = tree
    out = ["[", op]
    for child in children:
        out.extend(to_tokens(child))
    out.append("]")
    return out


def make_example(max_depth: int, max_args: int, min_tokens: int, max_tokens: int) -> tuple[str, int]:
    for _ in range(10_000):
        tree = generate_tree(1, max_depth, max_args)
        tokens = to_tokens(tree)
        if min_tokens <= len(tokens) <= max_tokens:
            return " ".join(tokens), to_value(tree)
    raise RuntimeError("failed to generate a ListOps example within length bounds")


def write_split(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["Source", "Target"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-args", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    total = args.train_samples + args.val_samples + args.test_samples
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    while len(rows) < total:
        source, target = make_example(args.max_depth, args.max_args, args.min_tokens, args.max_tokens)
        if source in seen:
            continue
        seen.add(source)
        rows.append((source, target))
        if len(rows) % 500 == 0:
            print(f"generated={len(rows)}", flush=True)

    train = rows[: args.train_samples]
    val = rows[args.train_samples : args.train_samples + args.val_samples]
    test = rows[args.train_samples + args.val_samples :]

    write_split(args.output_dir / "basic_train.tsv", train)
    write_split(args.output_dir / "basic_val.tsv", val)
    write_split(args.output_dir / "basic_test.tsv", test)

    lengths = [len(source.split()) for source, _ in rows]
    print(
        {
            "status": "ok",
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": float(np.mean(lengths)),
            "output_dir": str(args.output_dir),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
