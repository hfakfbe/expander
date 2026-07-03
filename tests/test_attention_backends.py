from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.graph.generation import build_layer_graphs
from src.graph.structures import apply_causal
from src.model.attention import apply_memory_routes
from src.model.backends import build_backend_bundle
from src.model.transformer import SequenceTransformer, model_config_from_resolved


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
            "local_window_size": 2,
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
            "random_memory": {
                "degree": 3,
                "density": None,
                "route_stride": 4,
                "route_multiplicity": 1,
                "memory_mode": "memory_replace",
                "memory_scale": 1.0,
            },
            "zigzag_logm": {"use_multiplicity_logm": True},
            "zigzag_boolean": {"use_multiplicity_logm": False},
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
        for method in ["dense", "local", "random_regular", "random_memory", "zigzag_logm", "zigzag_boolean"]:
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

    def test_sliding_window_local_mask(self) -> None:
        graph = build_layer_graphs(tiny_config("local"), torch.device("cpu"))[0]
        self.assertEqual(torch.nonzero(graph.mask[5], as_tuple=False).flatten().tolist(), [3, 4, 5, 6, 7])
        config = tiny_config("local")
        config["attention"]["causal"] = True
        graph = build_layer_graphs(config, torch.device("cpu"))[0]
        self.assertEqual(torch.nonzero(graph.mask[5], as_tuple=False).flatten().tolist(), [3, 4, 5])

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
        self.assertIsNone(boolean.log_m)
        self.assertEqual(max(value for counts in boolean.counts for value in counts.values()), 1)

    def test_random_memory_routes_are_applied(self) -> None:
        graph = build_layer_graphs(tiny_config("random_memory"), torch.device("cpu"))[0]
        self.assertIsNotNone(graph.memory_routes)
        hidden = torch.arange(1 * 16 * 2, dtype=torch.float32).reshape(1, 16, 2)
        routed = apply_memory_routes(hidden, graph.memory_routes)
        self.assertTrue(torch.equal(routed[:, 4, :], hidden[:, 0, :]))
        self.assertTrue(torch.equal(routed[:, 15, :], hidden[:, 11, :]))

    def test_missing_reused_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config("random_regular")
            config["attention"]["graph_artifact_policy"] = "reuse"
            config["attention"]["graph_artifact_root"] = str(Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                build_layer_graphs(config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

