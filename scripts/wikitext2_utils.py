from __future__ import annotations

import hashlib
import json
import math
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from graph_structures import load_graph_artifact
from synthetic_mvp import (
    Transformer,
    make_attention_artifacts,
    method_certification_fields,
    resolve_attention_backend,
)


class ByteTokenizer:
    name = "byte_level_utf8"
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        tokens = [int(byte) + 2 for byte in text.encode("utf-8")]
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens


@dataclass
class LMBatch:
    tokens: torch.Tensor
    loss_positions: torch.Tensor
    targets: torch.Tensor
    sequence_length: int
    T: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_string() -> str:
    return shlex.join([sys.executable, *sys.argv])


def write_command(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command_string() + "\n", encoding="utf-8")


def load_texts(dataset_dir: Path, split: str) -> list[str]:
    path = dataset_dir / f"{split}.jsonl"
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(str(row.get("text", "")))
    return texts


def build_blocks(dataset_dir: Path, split: str, sequence_length: int, tokenizer: ByteTokenizer) -> torch.Tensor:
    texts = [text for text in load_texts(dataset_dir, split) if text.strip()]
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_eos=True))
    block_len = int(sequence_length) + 1
    usable = (len(tokens) // block_len) * block_len
    if usable < block_len:
        raise ValueError(f"not enough tokens for split={split!r}, sequence_length={sequence_length}")
    data = torch.tensor(tokens[:usable], dtype=torch.long).view(-1, block_len)
    return data


def make_lm_batch(
    blocks: torch.Tensor,
    batch_size: int,
    batch_index: int,
    device: torch.device,
    sequence_length: int,
    T: int,
    seed: int,
    stream: str,
) -> LMBatch:
    if T < sequence_length:
        raise ValueError("attention length T must be >= sequence_length")
    gen = torch.Generator(device="cpu")
    digest = hashlib.sha256(f"{seed}|{stream}|{batch_index}|{len(blocks)}".encode("utf-8")).hexdigest()
    gen.manual_seed(int(digest[:16], 16) % (2**63 - 1))
    indices = torch.randint(0, len(blocks), (batch_size,), generator=gen)
    selected = blocks[indices]
    tokens = torch.full((batch_size, T), ByteTokenizer.pad_token_id, dtype=torch.long)
    tokens[:, :sequence_length] = selected[:, :sequence_length]
    targets = selected[:, 1 : sequence_length + 1]
    loss_positions = torch.arange(sequence_length, dtype=torch.long)
    return LMBatch(
        tokens=tokens.to(device),
        loss_positions=loss_positions.to(device),
        targets=targets.to(device),
        sequence_length=sequence_length,
        T=T,
    )


def lm_loss_and_metrics(logits: torch.Tensor, batch: LMBatch) -> tuple[torch.Tensor, dict]:
    selected_logits = logits[:, batch.loss_positions, :]
    loss = F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        batch.targets.reshape(-1),
    )
    pred = selected_logits.argmax(dim=-1)
    accuracy = float((pred == batch.targets).float().mean().item())
    return loss, {"token_accuracy": accuracy}


def build_runtime(config: dict, output_dir: Path, device: torch.device) -> SimpleNamespace:
    task = config["task"]
    model = config["model"]
    attention = config["attention"]
    train = config["train"]
    graph_artifact = load_graph_artifact(attention["graph_artifact"])
    sequence_length = int(task["sequence_length"])
    T = int(graph_artifact["T"])
    return SimpleNamespace(
        version=str(config.get("version", "v06")),
        task="wikitext2",
        dataset_dir=Path(task["dataset_dir"]),
        sequence_length=sequence_length,
        T=T,
        methods=[str(method) for method in attention["methods"]],
        block_size=int(graph_artifact["B"]),
        degree=int(graph_artifact["d"]),
        causal=bool(attention.get("causal", True)),
        graph_config=graph_artifact,
        graph_artifact=graph_artifact,
        graph_artifact_path=str(attention["graph_artifact"]),
        graph_certificate=dict(graph_artifact.get("certificate", {})),
        graph_id=str(graph_artifact.get("graph_id", "")),
        graph_seed=graph_artifact.get("graph_seed", ""),
        multiplicity_mode=str(attention.get("multiplicity", {}).get("mode", "boolean")),
        architecture=model.get("architecture", "transformer"),
        layers=int(model["layers"]),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        ffn_dim=int(model["ffn_dim"]),
        dropout=float(model["dropout"]),
        attention_backend=model["attention_backend"],
        steps=int(train.get("steps", 0)),
        batch_size=int(train["batch_size"]),
        eval_batches=int(train.get("eval_batches", 1)),
        learning_rate=float(train.get("learning_rate", 0.001)),
        log_every=int(train.get("log_every", 50)),
        eval_every=int(train.get("eval_every", 100)),
        seed=int(train.get("seed", train.get("seeds", [0])[0])),
        output_dir=output_dir,
        device=device,
        config_snapshot=config,
    )


def build_model_and_artifacts(args: SimpleNamespace, method: str, tokenizer: ByteTokenizer, device: torch.device):
    backend = resolve_attention_backend(args.attention_backend, method)
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        output_size=tokenizer.vocab_size,
        seq_len=args.T,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attention_backend=backend,
        block_size=args.block_size,
    ).to(device)
    artifacts = make_attention_artifacts(method, args.T, args, device, backend)
    return model, artifacts, backend


def run_eval_batches(
    model,
    artifacts,
    blocks: torch.Tensor,
    args: SimpleNamespace,
    tokenizer: ByteTokenizer,
    device: torch.device,
    split: str,
) -> dict:
    del tokenizer
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index in range(args.eval_batches):
            batch = make_lm_batch(
                blocks,
                args.batch_size,
                batch_index,
                device,
                args.sequence_length,
                args.T,
                args.seed,
                split,
            )
            logits = model(
                batch.tokens,
                artifacts.mask,
                artifacts.local_valid,
                artifacts.neighbors,
                artifacts.valid_neighbors,
                artifacts.block_pair_index,
                artifacts.local_log_m,
                artifacts.neighbor_log_m,
            )
            loss, metrics = lm_loss_and_metrics(logits, batch)
            total_loss += float(loss.item()) * batch.targets.numel()
            total_correct += metrics["token_accuracy"] * batch.targets.numel()
            total_tokens += batch.targets.numel()
    elapsed = max(time.perf_counter() - started, 1e-9)
    loss = total_loss / max(total_tokens, 1)
    return {
        f"{split}_loss": loss,
        f"{split}_perplexity": math.exp(min(loss, 20.0)),
        f"{split}_token_accuracy": total_correct / max(total_tokens, 1),
        f"{split}_tokens_per_sec": total_tokens / elapsed,
    }
