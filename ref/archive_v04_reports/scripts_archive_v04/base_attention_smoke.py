import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_synthetic_module():
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    import synthetic_mvp

    return synthetic_mvp


class BertSparseSelfAttention(nn.Module):
    def __init__(self, dense_attention: nn.Module, method: str, block_size: int, degree: int, backend: str, seed: int):
        super().__init__()
        self.num_attention_heads = dense_attention.num_attention_heads
        self.attention_head_size = dense_attention.attention_head_size
        self.all_head_size = dense_attention.all_head_size
        self.query = dense_attention.query
        self.key = dense_attention.key
        self.value = dense_attention.value
        self.dropout = dense_attention.dropout
        self.method = method
        self.block_size = block_size
        self.degree = degree
        self.backend = backend
        self.seed = seed

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise ValueError("sparse smoke replacement only supports encoder self-attention")
        synthetic = load_synthetic_module()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        seq_len = hidden_states.shape[1]
        if seq_len % self.block_size != 0:
            raise ValueError("sequence length must be divisible by block size")
        device = hidden_states.device
        mask = synthetic.build_attention_mask(
            self.method, seq_len, self.block_size, self.degree, device, self.seed
        )
        if self.backend == "dense_mask":
            context = synthetic.dense_attention(query_layer, key_layer, value_layer, mask)
        elif self.backend == "split":
            cross = synthetic.build_cross_mask(
                self.method, seq_len, self.block_size, self.degree, device, self.seed
            )
            cross_neighbors, valid_cross_neighbors = synthetic.mask_to_neighbors(cross)
            context = synthetic.local_cross_attention(
                query_layer, key_layer, value_layer, self.block_size, cross_neighbors, valid_cross_neighbors, self.dropout
            )
        elif self.backend == "blockpair":
            cross = synthetic.build_cross_mask(
                self.method, seq_len, self.block_size, self.degree, device, self.seed
            )
            cross_neighbors, valid_cross_neighbors = synthetic.mask_to_neighbors(cross)
            block_pair_index = synthetic.cross_neighbors_to_block_pair_index(
                cross_neighbors, valid_cross_neighbors, self.block_size
            )
            context = synthetic.local_blockpair_attention(
                query_layer,
                key_layer,
                value_layer,
                self.block_size,
                cross_neighbors,
                valid_cross_neighbors,
                block_pair_index,
                self.dropout,
            )
        else:
            raise ValueError(f"unsupported backend: {self.backend}")
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.shape[0], seq_len, self.all_head_size)
        outputs = (context,)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs


def patch_bert_attention(model, method: str, block_size: int, degree: int, backend: str, seed: int) -> int:
    count = 0
    for layer in model.bert.encoder.layer:
        layer.attention.self = BertSparseSelfAttention(
            layer.attention.self, method, block_size, degree, backend, seed
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("code/lra-benchmarks"))
    parser.add_argument("--task", choices=["listops", "imdb"], default="listops")
    parser.add_argument("--method", choices=["dense", "local", "random", "zigzag"], default="zigzag")
    parser.add_argument("--backend", choices=["dense_mask", "split", "blockpair"], default="split")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/base_attention_smoke/summary.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    support_dir = Path(__file__).resolve().parent / "smoke_support"
    sys.path.insert(0, str(support_dir))
    sys.path.insert(0, str(args.repo_dir.resolve()))
    os.chdir(args.repo_dir.resolve())

    from lra_config import get_listops_config, get_text_classification_config
    from lra_datasets import ImdbDataset, ListOpsDataset
    from run_model import get_model, transformers_collator

    if args.task == "listops":
        config, model_config = get_listops_config()
        dataset_cls = ListOpsDataset
    else:
        config, model_config = get_text_classification_config()
        dataset_cls = ImdbDataset
    config.max_length = args.seq_len
    config.batch_size = args.batch_size
    model_config.max_position_embeddings = args.seq_len
    model_config.hidden_size = args.hidden_size
    model_config.num_hidden_layers = args.layers
    model_config.num_attention_heads = args.heads
    model_config.intermediate_size = args.hidden_size * 4

    dataset = dataset_cls(config, split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"{args.task} dataset has no training examples")
    samples = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    inputs, target = transformers_collator(samples)
    model = get_model(config, model_config)
    patched_layers = patch_bert_attention(model, args.method, args.block_size, args.degree, args.backend, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    target = target.to(device)
    model.train()
    outputs = model(**inputs)
    loss = F.cross_entropy(outputs.logits, target)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    loss.backward()
    result = {
        "status": "ok",
        "task": args.task,
        "method": args.method,
        "backend": args.backend,
        "patched_layers": patched_layers,
        "logits_shape": list(outputs.logits.shape),
        "loss": float(loss.item()),
        "device": str(device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
