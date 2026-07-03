from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.config.loading import resolve_config
from src.tasks.common import JsonlDataset, load_split, task_spec_from_config
from src.tasks.registry import get_batcher, get_loss


def first_rows(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == count:
                    break
    return rows


class TaskContractTests(unittest.TestCase):
    def test_copy_batch_loss_and_metrics(self) -> None:
        config = resolve_config(Path("configs/runs/copy_dense.json"), requested_task="copy")
        spec = task_spec_from_config(config)
        rows = first_rows(Path("datasets/copy/train.jsonl"), 2)
        batch = get_batcher("copy")(rows, spec, torch.device("cpu"))
        self.assertEqual(tuple(batch.tokens.shape), (2, 2048))
        self.assertEqual(batch.target_positions[0, :3].tolist(), [1024, 1025, 1026])
        self.assertTrue(torch.equal(batch.tokens[:, :1024], batch.targets))
        logits = torch.randn(2, 2048, 64)
        loss, metrics = get_loss("copy")(logits, None, batch, spec)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("copy_token_accuracy", metrics)

    def test_selective_copy_tail_readout(self) -> None:
        config = resolve_config(Path("configs/runs/selective_copy_dense.json"), requested_task="selective_copy")
        spec = task_spec_from_config(config)
        rows = first_rows(Path("datasets/selective_copy/train.jsonl"), 2)
        batch = get_batcher("selective_copy")(rows, spec, torch.device("cpu"))
        self.assertEqual(batch.target_positions[0].tolist(), list(range(4112, 4128)))
        logits = torch.randn(2, 4128, 16)
        loss, metrics = get_loss("selective_copy")(logits, None, batch, spec)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("selective_copy_sequence_accuracy", metrics)

    def test_induction_associative_recall_positions(self) -> None:
        config = resolve_config(Path("configs/runs/induction_associative_recall_dense.json"), requested_task="induction_associative_recall")
        spec = task_spec_from_config(config)
        rows = first_rows(Path("datasets/induction_associative_recall/train.jsonl"), 2)
        batch = get_batcher("induction_associative_recall")(rows, spec, torch.device("cpu"))
        expected = [int(item["position"]) for item in rows[0]["target"]]
        self.assertEqual(batch.target_positions[0, : len(expected)].tolist(), expected)
        logits = torch.randn(2, 64, 8192)
        loss, metrics = get_loss("induction_associative_recall")(logits, None, batch, spec)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("position_value_token_accuracy", metrics)

    def test_listops_is_classification(self) -> None:
        config = resolve_config(Path("configs/runs/lra_listops_dense.json"), requested_task="lra_listops")
        spec = task_spec_from_config(config)
        rows = first_rows(Path("datasets/lra_listops/train.jsonl"), 2)
        batch = get_batcher("lra_listops")(rows, spec, torch.device("cpu"))
        self.assertIsNone(batch.target_positions)
        self.assertIsNotNone(batch.class_targets)
        logits = torch.randn(2, 10)
        loss, metrics = get_loss("lra_listops")(torch.randn(2, 5995, 10), logits, batch, spec)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("listops_accuracy", metrics)

    def test_full_dataset_iterator_does_not_skip_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            with path.open("w", encoding="utf-8") as fp:
                for index in range(7):
                    fp.write(json.dumps({"id": index}) + "\n")
            dataset = JsonlDataset(path)
            seen = [row["id"] for batch in dataset.batches(3, shuffle=False, seed=0) for row in batch]
            self.assertEqual(seen, list(range(7)))
            shuffled_a = [row["id"] for batch in dataset.batches(2, shuffle=True, seed=5, buffer_size=3) for row in batch]
            shuffled_b = [row["id"] for batch in dataset.batches(2, shuffle=True, seed=5, buffer_size=3) for row in batch]
            self.assertEqual(shuffled_a, shuffled_b)
            self.assertEqual(sorted(shuffled_a), list(range(7)))


if __name__ == "__main__":
    unittest.main()

