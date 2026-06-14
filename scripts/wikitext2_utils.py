from __future__ import annotations

import hashlib
import json
import math
import shutil
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from graph_structures import load_graph_artifact
from v07_artifacts import materialize_graph_artifact
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


class HFTokenizerWrapper:
    def __init__(self, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self.name = "byte_level_bpe"
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.eos_token_id = int(config.get("eos_token_id", 1))
        self.unk_token_id = int(config.get("unk_token_id", 2))
        self.vocab_size = int(config.get("vocab_size", tokenizer.get_vocab_size()))

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [int(v) for v in self.tokenizer.encode(text).ids]
        if not add_eos and ids and ids[-1] == self.eos_token_id:
            return ids[:-1]
        return ids


def load_phase4_tokenizer(tokenizer_dir: Path) -> HFTokenizerWrapper:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("tokenizers package is required to read Phase 4 tokenizer.json") from exc
    config = read_json(tokenizer_dir / "tokenizer_config.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    return HFTokenizerWrapper(tokenizer, config)


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
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
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


def load_tokenized_blocks(path: Path) -> torch.Tensor:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append([int(v) for v in row["input_ids"]])
    if not rows:
        raise ValueError(f"no tokenized blocks in {path}")
    return torch.tensor(rows, dtype=torch.long)


def make_lm_batch(
    blocks: torch.Tensor,
    batch_size: int,
    batch_index: int,
    device: torch.device,
    sequence_length: int,
    T: int,
    seed: int,
    stream: str,
    pad_token_id: int = ByteTokenizer.pad_token_id,
    eos_token_id: int = ByteTokenizer.eos_token_id,
) -> LMBatch:
    if T < sequence_length:
        raise ValueError("attention length T must be >= sequence_length")
    gen = torch.Generator(device="cpu")
    digest = hashlib.sha256(f"{seed}|{stream}|{batch_index}|{len(blocks)}".encode("utf-8")).hexdigest()
    gen.manual_seed(int(digest[:16], 16) % (2**63 - 1))
    indices = torch.randint(0, len(blocks), (batch_size,), generator=gen)
    selected = blocks[indices]
    tokens = torch.full((batch_size, T), int(pad_token_id), dtype=torch.long)
    tokens[:, :sequence_length] = selected[:, :sequence_length]
    if selected.shape[1] >= sequence_length + 1:
        targets = selected[:, 1 : sequence_length + 1]
    else:
        eos_column = torch.full((batch_size, 1), int(eos_token_id), dtype=torch.long)
        targets = torch.cat([selected[:, 1:sequence_length], eos_column], dim=1)
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
    graph_materialization = None
    if config.get("graph") and not bool(config.get("graph", {}).get("generate", False)):
        graph_materialization = materialize_graph_artifact(config, output_dir, require=True)
        attention["graph_artifact"] = str(graph_materialization.selected_graph_path)
    graph_artifact = graph_materialization.artifact if graph_materialization is not None else load_graph_artifact(attention["graph_artifact"])
    sequence_length = int(task["sequence_length"])
    T = int(graph_artifact["T"])
    eval_cfg = config.get("eval", {})
    train = config["train"]
    data_cfg = config.get("data", {})
    tokenizer_cfg = config.get("tokenizer", {})
    data_phase_dir = Path(task.get("data_phase_dir", data_cfg.get("source_dir", task.get("dataset_dir", ""))))
    tokenizer_source_dir = Path(tokenizer_cfg.get("source_dir", data_phase_dir / "artifacts/tokenizer"))
    tokenizer_path = Path(tokenizer_cfg.get("path", tokenizer_source_dir / "tokenizer.json"))
    train_path = Path(data_cfg.get("tokenized_train_path", data_phase_dir / "tokenized/train_blocks.jsonl"))
    test_path = Path(data_cfg.get("tokenized_test_path", data_phase_dir / "tokenized/test_blocks.jsonl"))
    if "*" in str(train_path):
        train_matches = sorted(train_path.parent.glob(train_path.name))
        if not train_matches:
            raise FileNotFoundError(f"no train tokenized files match {train_path}")
        train_path = train_matches[0]
    if "*" in str(test_path):
        test_matches = sorted(test_path.parent.glob(test_path.name))
        if not test_matches:
            raise FileNotFoundError(f"no test tokenized files match {test_path}")
        test_path = test_matches[0]
    return SimpleNamespace(
        version=str(config.get("version", "v06")),
        task="wikitext",
        dataset_dir=Path(task.get("dataset_dir", data_phase_dir)),
        data_phase_dir=data_phase_dir,
        tokenizer_source_dir=tokenizer_source_dir,
        tokenizer_path=tokenizer_path,
        tokenized_train_path=train_path,
        tokenized_test_path=test_path,
        data_readiness_path=Path(data_cfg.get("data_readiness_path", data_phase_dir / "data_readiness.json")),
        tokenization_summary_path=Path(
            data_cfg.get("tokenization_summary_path", data_phase_dir / "tokenization_summary.json")
        ),
        expected_tokenizer_sha256=tokenizer_cfg.get("expected_tokenizer_sha256", ""),
        expected_tokenized_train_sha256=data_cfg.get("expected_tokenized_train_sha256", ""),
        expected_tokenized_test_sha256=data_cfg.get("expected_tokenized_test_sha256", ""),
        require_tokenizer_sha256_match=bool(tokenizer_cfg.get("require_sha256_match", False)),
        require_data_sha256_match=bool(data_cfg.get("require_sha256_match", False)),
        sequence_length=sequence_length,
        T=T,
        methods=[str(method) for method in attention["methods"]],
        block_size=int(graph_artifact["B"]),
        degree=int(graph_artifact["d"]),
        causal=bool(attention.get("causal", True)),
        graph_config=graph_artifact,
        graph_artifact=graph_artifact,
        graph_artifact_path=str(attention["graph_artifact"]),
        graph_materialization=graph_materialization,
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
        epochs=int(train.get("epochs", 0)),
        steps=int(train.get("steps", 0)),
        batch_size=int(train["batch_size"]),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 1)),
        effective_batch_size=int(train.get("effective_batch_size", int(train["batch_size"]))),
        eval_batch_size=int(eval_cfg.get("batch_size", train.get("eval_batch_size", train["batch_size"]))),
        eval_batches=eval_cfg.get("eval_batches", train.get("eval_batches", 1)),
        learning_rate=float(train.get("learning_rate", 0.001)),
        base_learning_rate=float(train.get("base_learning_rate", train.get("learning_rate", 0.001))),
        lr_scheduler=str(train.get("lr_scheduler", train.get("default_lr_scheduler", "constant"))),
        warmup_ratio=float(train.get("warmup_ratio", 0.0)),
        min_lr_ratio=float(train.get("min_lr_ratio", 0.0)),
        method_overrides=dict(config.get("method_overrides", train.get("method_overrides", {}))),
        weight_decay=float(train.get("weight_decay", 0.0)),
        grad_clip_norm=float(train.get("grad_clip_norm", 0.0)),
        log_every=int(train.get("log_every", 50)),
        eval_every=int(train.get("eval_every", 100)),
        seed=int(train.get("seed", train.get("seeds", [0])[0])),
        output_dir=output_dir,
        device=device,
        config_snapshot=config,
    )


def copy_phase4_artifacts(args: SimpleNamespace, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dst = output_dir / "artifacts/tokenizer"
    tokenizer_dst.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "tokenizer_training.json"]:
        src = args.tokenizer_source_dir / name
        if src.exists():
            shutil.copyfile(src, tokenizer_dst / name)
    manifest = {
        "wikitext_data_phase_dir": str(args.data_phase_dir),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenized_train_path": str(args.tokenized_train_path),
        "tokenized_test_path": str(args.tokenized_test_path),
        "data_readiness_path": str(args.data_readiness_path),
        "tokenization_summary_path": str(args.tokenization_summary_path),
        "tokenizer_sha256": file_sha256(args.tokenizer_path) if args.tokenizer_path.exists() else "",
        "tokenized_train_sha256": file_sha256(args.tokenized_train_path) if args.tokenized_train_path.exists() else "",
        "tokenized_test_sha256": file_sha256(args.tokenized_test_path) if args.tokenized_test_path.exists() else "",
    }
    write_json(output_dir / "phase4_data_artifact_manifest.json", manifest)
    if args.data_readiness_path.exists():
        shutil.copyfile(args.data_readiness_path, output_dir / "data_readiness.json")
    if args.tokenization_summary_path.exists():
        shutil.copyfile(args.tokenization_summary_path, output_dir / "tokenization_summary.json")
    return manifest


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
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
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
