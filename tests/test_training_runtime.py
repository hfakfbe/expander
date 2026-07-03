from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.config.loading import apply_cli_overrides, resolve_config
from src.config.validation import ConfigError
from src.training.runner import run, train


def tiny_copy_rows(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(8):
        source = [(index + offset) % 8 + 1 for offset in range(4)]
        rows.append({"id": str(index), "input": source + [9, 9, 9, 9], "target": source, "task": "copy"})
    for split in ["train", "test"]:
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row) + "\n")


def tiny_config(tmp: Path, run_id: str, max_steps: int = 2) -> dict:
    data_root = tmp / "data"
    tiny_copy_rows(data_root)
    return {
        "task": {
            "name": "copy",
            "dataset_root": str(data_root),
            "train_split": "train",
            "eval_split": "test",
            "vocab_size": 10,
            "output_size": 10,
            "sequence_length": 8,
            "target_length": 4,
            "loss_type": "sequence",
            "pad_token_id": 0,
            "marker_token_id": 9,
            "source_length": 4,
        },
        "model": {
            "num_layers": 1,
            "dim": 8,
            "dim_ffn": 16,
            "num_heads": 2,
            "activation": "gelu",
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "norm_type": "layernorm",
            "positional_encoding": "learned_absolute",
            "rope": {"enabled": False, "theta": 10000.0},
        },
        "training": {
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "scheduler": "constant",
            "min_learning_rate": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "max_steps": max_steps,
            "epochs": None,
            "batch_size": 2,
            "minibatch_size": 2,
            "gradient_accumulation_steps": 1,
            "log_step": 1,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "seed": 123,
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "grad_clip_norm": 0.0,
            "resume_checkpoint": None,
            "shuffle_buffer_size": 4,
            "eval_batches": 1,
            "final_eval_batches": 1,
        },
        "attention": {
            "method": "local",
            "causal": False,
            "local_mode": "sliding_window",
            "local_window_size": 2,
            "include_local_edges": True,
            "per_layer_random": False,
            "graph_seed": 0,
            "per_layer_graph_seeds": None,
            "q": 2,
            "B": 4,
            "d": 2,
            "density": None,
            "graph_artifact_root": str(tmp / "graphs" / run_id),
            "graph_artifact_policy": "regenerate",
            "random_regular": {"degree": 2, "density": None},
            "zigzag_logm": {"use_multiplicity_logm": True},
            "zigzag_boolean": {"use_multiplicity_logm": False},
        },
        "memory_rollout": {
            "enabled": False,
            "alpha": 0.5,
            "injection_scale": 2.0,
            "head_merge": "mean",
            "update": "lazy",
            "initial_state": "input",
        },
        "run": {
            "output_root": str(tmp / "runs"),
            "run_id": run_id,
            "save_checkpoints": True,
            "save_graph_artifacts": True,
            "save_metrics": True,
            "save_manifest": True,
            "manifest_path": None,
            "config_sha256_required": None,
            "dataset_sha256_required": None,
            "resume_checkpoint": None,
        },
    }


class TrainingRuntimeTests(unittest.TestCase):
    def test_check_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config = tiny_config(tmp, "check", max_steps=1)
            result = run(config, "check", ["run_task", "--mode", "check"])
            run_dir = tmp / "runs" / "check"
            self.assertEqual(result["mode"], "check")
            self.assertTrue((run_dir / "resolved_config.json").exists())
            self.assertTrue((run_dir / "final_metrics.json").exists())
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_config"]["attention"]["method"], "local")
            self.assertIn("train", manifest["dataset_sha256"])
            self.assertTrue(manifest["graph_artifact_sha256"])

    def test_checkpoint_resume_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            full = tiny_config(tmp, "full", max_steps=2)
            train(full, tmp / "runs" / "full", ["run_task"])
            first = tiny_config(tmp, "first", max_steps=1)
            train(first, tmp / "runs" / "first", ["run_task"])
            resumed = tiny_config(tmp, "resumed", max_steps=2)
            resumed["training"]["resume_checkpoint"] = str(tmp / "runs" / "first" / "checkpoints" / "step_000001.pt")
            train(resumed, tmp / "runs" / "resumed", ["run_task"])
            full_state = torch.load(tmp / "runs" / "full" / "checkpoints" / "step_000002.pt", map_location="cpu")["model_state"]
            resumed_state = torch.load(tmp / "runs" / "resumed" / "checkpoints" / "step_000002.pt", map_location="cpu")["model_state"]
            for key in full_state:
                self.assertTrue(torch.allclose(full_state[key], resumed_state[key]), key)

    def test_epochs_limit_training_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config = tiny_config(tmp, "epochs", max_steps=99)
            config["training"]["epochs"] = 2
            result = train(config, tmp / "runs" / "epochs", ["run_task"])
            self.assertEqual(result["requested_max_steps"], 99)
            self.assertEqual(result["epoch_limited_steps"], 8)
            self.assertEqual(result["final_step"], 8)
            self.assertTrue((tmp / "runs" / "epochs" / "checkpoints" / "step_000008.pt").exists())
            self.assertFalse((tmp / "runs" / "epochs" / "checkpoints" / "step_000099.pt").exists())

    def test_config_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            path = tmp / "bad.json"
            path.write_text(json.dumps({"task": {"name": "copy"}}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(path, requested_task="copy")
            good_path = tmp / "good.json"
            good_path.write_text(json.dumps(tiny_config(tmp, "cfg", max_steps=1)), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(good_path, requested_task="selective_copy")
            with self.assertRaises(ConfigError):
                apply_cli_overrides(tiny_config(tmp, "cfg", max_steps=1), ["model.dim=64"])
            bad_method = tiny_config(tmp, "bad_method", max_steps=1)
            bad_method["attention"]["method"] = "random_memory"
            bad_method_path = tmp / "bad_method.json"
            bad_method_path.write_text(json.dumps(bad_method), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(bad_method_path, requested_task="copy")
            bad_alpha = tiny_config(tmp, "bad_alpha", max_steps=1)
            bad_alpha["memory_rollout"]["alpha"] = 1.5
            bad_alpha_path = tmp / "bad_alpha.json"
            bad_alpha_path.write_text(json.dumps(bad_alpha), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(bad_alpha_path, requested_task="copy")
            bad_epochs = tiny_config(tmp, "bad_epochs", max_steps=1)
            bad_epochs["training"]["epochs"] = 0
            bad_epochs_path = tmp / "bad_epochs.json"
            bad_epochs_path.write_text(json.dumps(bad_epochs), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(bad_epochs_path, requested_task="copy")
            bad_norm = tiny_config(tmp, "bad_norm", max_steps=1)
            bad_norm["model"]["norm_type"] = "batchnorm"
            bad_norm_path = tmp / "bad_norm.json"
            bad_norm_path.write_text(json.dumps(bad_norm), encoding="utf-8")
            with self.assertRaises(ConfigError):
                resolve_config(bad_norm_path, requested_task="copy")


if __name__ == "__main__":
    unittest.main()
