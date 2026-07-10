from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.graph.generation import build_layer_graphs
from src.graph.structures import apply_causal
from src.model.backends import build_backend_bundle
from src.model.transformer import MemoryRolloutConfig, SequenceTransformer, model_config_from_resolved, rollout_memory_update


def tiny_config(method: str) -> dict:
    return {
        "task": {
            "name": "copy",
            "dataset_root": "unused",
            "train_split": "train",
            "eval_split": "test",
            "vocab_size": 16,
            "output_size": 16,
            "sequence_length": 16,
            "target_length": 4,
            "loss_type": "sequence",
            "pad_token_id": 0,
            "marker_token_id": 15,
            "source_length": 8,
        },
        "model": {
            "num_layers": 2,
            "dim": 16,
            "dim_ffn": 32,
            "num_heads": 4,
            "activation": "gelu",
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "norm_type": "layernorm",
            "positional_encoding": "rope",
            "rope": {"enabled": True, "theta": 10000.0},
        },
        "training": {
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "scheduler": "constant",
            "min_learning_rate": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "max_steps": 1,
            "epochs": None,
            "batch_size": 2,
            "minibatch_size": 2,
            "gradient_accumulation_steps": 1,
            "log_step": 1,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "seed": 0,
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "grad_clip_norm": 0.0,
            "resume_checkpoint": None,
            "shuffle_buffer_size": 8,
            "eval_batches": 1,
            "final_eval_batches": 1,
        },
        "attention": {
            "method": method,
            "causal": False,
            "local_mode": "sliding_window",
            "local_window_size": 4,
            "include_local_edges": True,
            "per_layer_random": True,
            "graph_seed": 3,
            "per_layer_graph_seeds": None,
            "q": 4,
            "B": 4,
            "d": 2,
            "density": None,
            "graph_artifact_root": "unused",
            "graph_artifact_policy": "regenerate",
            "random_regular": {"degree": 3, "density": None},
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
            "output_root": "unused",
            "run_id": "unused",
            "save_checkpoints": False,
            "save_graph_artifacts": False,
            "save_metrics": True,
            "save_manifest": True,
            "manifest_path": None,
            "config_sha256_required": None,
            "dataset_sha256_required": None,
            "resume_checkpoint": None,
        },
    }


class AttentionBackendTests(unittest.TestCase):
    def test_each_method_has_real_forward_path(self) -> None:
        outputs = {}
        tokens = torch.arange(32, dtype=torch.long).reshape(2, 16) % 16
        pad_mask = torch.ones((2, 16), dtype=torch.bool)
        for method in ["dense", "local", "random_regular", "zigzag_logm", "zigzag_boolean"]:
            config = tiny_config(method)
            graphs = build_layer_graphs(config, torch.device("cpu"))
            bundle = build_backend_bundle(graphs, 0.0)
            model = SequenceTransformer(model_config_from_resolved(config))
            token_logits, _ = model(tokens, pad_mask, bundle)
            self.assertEqual(tuple(token_logits.shape), (2, 16, 16))
            self.assertEqual(bundle.backends[0].kind, method)
            outputs[method] = token_logits.detach()
        self.assertFalse(torch.allclose(outputs["dense"], outputs["local"]))

    def test_causal_mask_changes_behavior(self) -> None:
        config = tiny_config("dense")
        noncausal = build_layer_graphs(config, torch.device("cpu"))[0].mask
        config["attention"]["causal"] = True
        causal = build_layer_graphs(config, torch.device("cpu"))[0].mask
        self.assertGreater(int(noncausal.sum().item()), int(causal.sum().item()))
        self.assertEqual(torch.nonzero(causal[0], as_tuple=False).flatten().tolist(), [0])

    def test_class_pooling_modes_run(self) -> None:
        for pooling in ["mean", "last"]:
            config = tiny_config("dense")
            config["task"]["loss_type"] = "classification"
            config["task"]["output_size"] = 7
            config["model"]["class_pooling"] = pooling
            graphs = build_layer_graphs(config, torch.device("cpu"))
            bundle = build_backend_bundle(graphs, 0.0)
            model = SequenceTransformer(model_config_from_resolved(config))
            tokens = torch.arange(32, dtype=torch.long).reshape(2, 16) % 16
            pad_mask = torch.ones((2, 16), dtype=torch.bool)
            _, class_logits = model(tokens, pad_mask, bundle)
            self.assertIsNotNone(class_logits)
            self.assertEqual(tuple(class_logits.shape), (2, 7))

        config = tiny_config("dense")
        config["task"]["loss_type"] = "classification"
        config["model"]["class_pooling"] = "bad"
        with self.assertRaises(ValueError):
            SequenceTransformer(model_config_from_resolved(config))

    def test_sliding_window_local_mask(self) -> None:
        config = tiny_config("local")
        config["attention"]["local_window_size"] = 5
        graph = build_layer_graphs(config, torch.device("cpu"))[0]
        self.assertEqual(torch.nonzero(graph.mask[5], as_tuple=False).flatten().tolist(), [3, 4, 5, 6, 7])
        config["attention"]["causal"] = True
        graph = build_layer_graphs(config, torch.device("cpu"))[0]
        self.assertEqual(torch.nonzero(graph.mask[5], as_tuple=False).flatten().tolist(), [1, 2, 3, 4, 5])

    def test_per_layer_random_reproducible_and_distinct(self) -> None:
        config = tiny_config("random_regular")
        graphs_a = build_layer_graphs(config, torch.device("cpu"))
        graphs_b = build_layer_graphs(config, torch.device("cpu"))
        self.assertTrue(torch.equal(graphs_a[0].mask, graphs_b[0].mask))
        self.assertFalse(torch.equal(graphs_a[0].mask, graphs_a[1].mask))
        config["attention"]["per_layer_random"] = False
        graphs = build_layer_graphs(config, torch.device("cpu"))
        self.assertTrue(torch.equal(graphs[0].mask, graphs[1].mask))

    def test_zigzag_weight_semantics(self) -> None:
        logm = build_layer_graphs(tiny_config("zigzag_logm"), torch.device("cpu"))[0]
        boolean = build_layer_graphs(tiny_config("zigzag_boolean"), torch.device("cpu"))[0]
        self.assertIsNotNone(logm.log_m)
        self.assertGreater(float(logm.log_m.max().item()), 0.0)
        self.assertGreater(max(value for counts in logm.counts for value in counts.values()), 1)
        self.assertIsNone(boolean.log_m)
        self.assertEqual(max(value for counts in boolean.counts for value in counts.values()), 1)
        self.assertTrue(torch.equal(logm.mask, boolean.mask))

    def test_zigzag_uses_seeded_random_permutations(self) -> None:
        for method in ["zigzag_logm", "zigzag_boolean"]:
            config = tiny_config(method)
            graphs_a = build_layer_graphs(config, torch.device("cpu"))
            graphs_b = build_layer_graphs(config, torch.device("cpu"))
            self.assertTrue(torch.equal(graphs_a[0].mask, graphs_b[0].mask))
            self.assertFalse(torch.equal(graphs_a[0].mask, graphs_a[1].mask))
            config["attention"]["per_layer_random"] = False
            graphs = build_layer_graphs(config, torch.device("cpu"))
            self.assertTrue(torch.equal(graphs[0].mask, graphs[1].mask))

    def test_memory_rollout_update_matches_formula(self) -> None:
        memory = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
        head_a = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.75, 0.0],
                [0.0, 0.5, 0.5],
            ]
        )
        head_b = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 0.5],
            ]
        )
        transition = torch.stack([head_a, head_b], dim=0).unsqueeze(0)
        config = MemoryRolloutConfig(
            enabled=True,
            alpha=0.5,
            injection_scale=2.0,
            head_merge="mean",
            update="lazy",
            initial_state="input",
        )
        expected = 0.5 * memory + 0.5 * torch.matmul((head_a + head_b).unsqueeze(0) / 2.0, memory)
        self.assertTrue(torch.allclose(rollout_memory_update(memory, transition, config), expected))

    def test_memory_rollout_runs_with_zigzag(self) -> None:
        config = tiny_config("zigzag_logm")
        config["memory_rollout"]["enabled"] = True
        graphs = build_layer_graphs(config, torch.device("cpu"))
        bundle = build_backend_bundle(graphs, 0.0)
        model = SequenceTransformer(model_config_from_resolved(config))
        tokens = torch.arange(32, dtype=torch.long).reshape(2, 16) % 16
        pad_mask = torch.ones((2, 16), dtype=torch.bool)
        token_logits, _, memory_state = model(tokens, pad_mask, bundle, return_memory=True)
        self.assertEqual(tuple(token_logits.shape), (2, 16, 16))
        self.assertIsNotNone(memory_state)
        self.assertEqual(tuple(memory_state.shape), (2, 16, 16))
        self.assertEqual(bundle.backends[0].kind, "zigzag_logm")

    def test_rmsnorm_forward_path(self) -> None:
        config = tiny_config("local")
        config["model"]["norm_type"] = "rmsnorm"
        graphs = build_layer_graphs(config, torch.device("cpu"))
        bundle = build_backend_bundle(graphs, 0.0)
        model = SequenceTransformer(model_config_from_resolved(config))
        tokens = torch.arange(32, dtype=torch.long).reshape(2, 16) % 16
        pad_mask = torch.ones((2, 16), dtype=torch.bool)
        token_logits, _ = model(tokens, pad_mask, bundle)
        self.assertEqual(tuple(token_logits.shape), (2, 16, 16))

    def test_missing_reused_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config("random_regular")
            config["attention"]["graph_artifact_policy"] = "reuse"
            config["attention"]["graph_artifact_root"] = str(Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                build_layer_graphs(config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
