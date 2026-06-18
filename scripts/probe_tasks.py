from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from synthetic_mvp_core.model import Block


PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
UNK_TOKEN_ID = 2


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


class ProbeEncoder:
    def __init__(self, payload: dict):
        self.payload = dict(payload)
        self.encoder_type = str(payload["encoder_type"])
        self.pad_token_id = int(payload.get("pad_token_id", PAD_TOKEN_ID))
        eos_value = payload.get("eos_token_id", EOS_TOKEN_ID)
        unk_value = payload.get("unk_token_id", UNK_TOKEN_ID)
        self.eos_token_id = None if eos_value is None else int(eos_value)
        self.unk_token_id = None if unk_value is None else int(unk_value)
        self.token_to_id = {str(k): int(v) for k, v in payload.get("token_to_id", {}).items()}
        self.vocab_size = int(payload["vocab_size"])

    def encode_input(self, value: Any, add_eos: bool = False) -> list[int]:
        if self.encoder_type == "byte_utf8":
            text = value if isinstance(value, str) else target_to_text(value)
            tokens = [int(byte) + 3 for byte in text.encode("utf-8")]
        elif self.encoder_type == "listops_vocab":
            tokens = [self.token_to_id.get(str(item), self.unk_token_id) for item in list(value)]
        elif self.encoder_type == "integer_shift":
            tokens = [int(item) + 1 for item in list(value)]
        elif self.encoder_type == "identity_integer":
            tokens = [int(item) for item in list(value)]
        else:
            raise ValueError(f"unknown encoder_type={self.encoder_type!r}")
        if add_eos:
            if self.eos_token_id is None:
                raise ValueError(f"encoder_type={self.encoder_type!r} has no EOS token")
            tokens.append(self.eos_token_id)
        return tokens

    def encode_target(self, value: Any, add_eos: bool = False) -> list[int]:
        if self.encoder_type == "byte_utf8":
            return self.encode_input(target_to_text(value), add_eos=add_eos)
        if self.encoder_type == "listops_vocab":
            if isinstance(value, list):
                return [self.token_to_id.get(str(item), self.unk_token_id) for item in value]
            return [int(value)]
        if self.encoder_type == "integer_shift":
            if isinstance(value, list):
                return [int(item) + 1 for item in value]
            return [int(value) + 1]
        if self.encoder_type == "identity_integer":
            if isinstance(value, list):
                return [int(item) for item in value]
            return [int(value)]
        raise ValueError(f"unknown encoder_type={self.encoder_type!r}")

    def decode(self, values: list[int]) -> str:
        if self.encoder_type == "byte_utf8":
            raw = bytes(max(0, int(value) - 3) for value in values if int(value) >= 3)
            return raw.decode("utf-8", errors="replace")
        inv = {value: key for key, value in self.token_to_id.items()}
        if inv:
            return " ".join(inv.get(int(value), "<unk>") for value in values)
        if self.encoder_type == "identity_integer":
            return " ".join(str(int(value)) for value in values)
        return " ".join(str(int(value) - 1) for value in values)


def target_to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def write_encoder(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_encoder(path: Path) -> ProbeEncoder:
    return ProbeEncoder(json.loads(path.read_text(encoding="utf-8")))


def build_listops_encoder(train_path: Path, output_path: Path) -> dict:
    values = {"<pad>": PAD_TOKEN_ID, "<eos>": EOS_TOKEN_ID, "<unk>": UNK_TOKEN_ID}
    with train_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            for token in row["input"]:
                token = str(token)
                if token not in values:
                    values[token] = len(values)
    payload = {
        "encoder_type": "listops_vocab",
        "token_to_id": values,
        "vocab_size": len(values),
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
    }
    write_encoder(output_path, payload)
    return payload


def integer_encoder(max_value: int, output_path: Path) -> dict:
    payload = {
        "encoder_type": "integer_shift",
        "vocab_size": int(max_value) + 2,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "integer_shift": 1,
    }
    write_encoder(output_path, payload)
    return payload


def identity_integer_encoder(vocab_size: int, output_path: Path) -> dict:
    payload = {
        "encoder_type": "identity_integer",
        "vocab_size": int(vocab_size),
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": None,
        "unk_token_id": None,
        "integer_shift": 0,
    }
    write_encoder(output_path, payload)
    return payload


def byte_encoder(output_path: Path) -> dict:
    payload = {
        "encoder_type": "byte_utf8",
        "vocab_size": 259,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "unk_token_id": UNK_TOKEN_ID,
        "byte_offset": 3,
    }
    write_encoder(output_path, payload)
    return payload


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        offset = 0
        with path.open("rb") as fp:
            for line in fp:
                self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"empty JSONL split: {path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def row(self, index: int) -> dict:
        index = int(index) % len(self.offsets)
        with self.path.open("r", encoding="utf-8") as fp:
            fp.seek(self.offsets[index])
            return json.loads(fp.readline())

    def sample(self, batch_size: int, seed: int, stream: str, step: int, limit: int | None = None) -> list[dict]:
        n = min(len(self.offsets), int(limit)) if limit else len(self.offsets)
        rng = random.Random(stable_seed(seed, stream, step, self.path, n))
        return [self.row(rng.randrange(n)) for _ in range(batch_size)]

    def batches(self, batch_size: int, limit: int | None = None) -> Iterator[list[dict]]:
        n = min(len(self.offsets), int(limit)) if limit else len(self.offsets)
        for start in range(0, n, batch_size):
            yield [self.row(index) for index in range(start, min(start + batch_size, n))]


@dataclass
class ProbeBatch:
    tokens: torch.Tensor
    target_positions: torch.Tensor | None
    targets: torch.Tensor | None
    target_mask: torch.Tensor | None
    class_targets: torch.Tensor | None
    pad_mask: torch.Tensor
    subtasks: list[str]
    example_count: int
    token_count: int


def _target_entries(row: dict) -> tuple[list[int], list[int]]:
    positions = []
    values = []
    for item in row.get("target", []):
        if isinstance(item, dict) and "position" in item and "value" in item:
            positions.append(int(item["position"]))
            values.append(int(item["value"]) + 1)
    return positions, values


def make_probe_batch(rows: list[dict], task_record: dict, encoder: ProbeEncoder, device: torch.device) -> ProbeBatch:
    if bool(task_record.get("copy_corrected_v01", False)):
        return make_copy_corrected_batch(rows, task_record, encoder, device)
    loss_type = str(task_record["resolved_loss_type"])
    input_limit = int(task_record["resolved_runtime_input_length"])
    target_limit = int(task_record["resolved_runtime_target_length"])
    T = int(task_record["resolved_padded_sequence_length"])
    readout_start = int(task_record["resolved_readout_start"])
    tokens = torch.full((len(rows), T), encoder.pad_token_id, dtype=torch.long)
    pad_mask = torch.zeros((len(rows), T), dtype=torch.bool)
    subtasks = []

    target_positions = None
    targets = None
    target_mask = None
    class_targets = None

    if loss_type in {"sequence_cross_entropy", "retrieval_sequence_cross_entropy"}:
        target_positions = torch.zeros((len(rows), target_limit), dtype=torch.long)
        targets = torch.zeros((len(rows), target_limit), dtype=torch.long)
        target_mask = torch.zeros((len(rows), target_limit), dtype=torch.bool)
    elif loss_type == "mqar_position_cross_entropy":
        max_targets = max(1, min(target_limit, max((len(row.get("target", [])) for row in rows), default=1)))
        target_positions = torch.zeros((len(rows), max_targets), dtype=torch.long)
        targets = torch.zeros((len(rows), max_targets), dtype=torch.long)
        target_mask = torch.zeros((len(rows), max_targets), dtype=torch.bool)
    elif loss_type == "classification_cross_entropy":
        class_targets = torch.zeros((len(rows),), dtype=torch.long)
    else:
        raise ValueError(f"unsupported loss_type={loss_type!r}")

    token_count = 0
    for batch_index, row in enumerate(rows):
        input_ids = encoder.encode_input(row.get("input"), add_eos=False)[:input_limit]
        if input_ids:
            tokens[batch_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            pad_mask[batch_index, : len(input_ids)] = True
            token_count += len(input_ids)
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        subtasks.append(str(meta.get("ruler_task") or row.get("variant") or "not_applicable"))
        if loss_type in {"sequence_cross_entropy", "retrieval_sequence_cross_entropy"}:
            assert target_positions is not None and targets is not None and target_mask is not None
            target_ids = encoder.encode_target(row.get("target"), add_eos=encoder.encoder_type == "byte_utf8")[:target_limit]
            for offset, target_id in enumerate(target_ids):
                pos = readout_start + offset
                if pos >= T:
                    break
                target_positions[batch_index, offset] = pos
                targets[batch_index, offset] = int(target_id)
                target_mask[batch_index, offset] = True
        elif loss_type == "mqar_position_cross_entropy":
            assert target_positions is not None and targets is not None and target_mask is not None
            positions, values = _target_entries(row)
            for offset, (pos, value) in enumerate(zip(positions[: target_positions.shape[1]], values[: target_positions.shape[1]])):
                if pos >= T:
                    continue
                target_positions[batch_index, offset] = pos
                targets[batch_index, offset] = value
                target_mask[batch_index, offset] = True
        elif loss_type == "classification_cross_entropy":
            assert class_targets is not None
            class_targets[batch_index] = int(row.get("target"))

    return ProbeBatch(
        tokens=tokens.to(device),
        target_positions=target_positions.to(device) if target_positions is not None else None,
        targets=targets.to(device) if targets is not None else None,
        target_mask=target_mask.to(device) if target_mask is not None else None,
        class_targets=class_targets.to(device) if class_targets is not None else None,
        pad_mask=pad_mask.to(device),
        subtasks=subtasks,
        example_count=len(rows),
        token_count=token_count,
    )


def make_copy_corrected_batch(rows: list[dict], task_record: dict, encoder: ProbeEncoder, device: torch.device) -> ProbeBatch:
    if encoder.encoder_type != "identity_integer":
        raise ValueError("copy_corrected_v01 requires identity_integer encoder")
    batch_size = len(rows)
    input_len = int(task_record.get("resolved_runtime_input_length", 2048))
    target_len = int(task_record.get("resolved_runtime_target_length", 1024))
    padded_len = int(task_record.get("resolved_padded_sequence_length", 2048))
    marker_id = int(task_record.get("marker_token_id", 63))
    if (input_len, target_len, padded_len) != (2048, 1024, 2048):
        raise ValueError(
            "copy_corrected_v01 requires input/target/T=(2048,1024,2048), "
            f"got {(input_len, target_len, padded_len)}"
        )
    tokens = torch.empty((batch_size, 2048), dtype=torch.long)
    targets = torch.empty((batch_size, 1024), dtype=torch.long)
    target_positions = torch.arange(1024, 2048, dtype=torch.long).repeat(batch_size, 1)
    target_mask = torch.ones((batch_size, 1024), dtype=torch.bool)
    valid_token_mask = torch.ones((batch_size, 2048), dtype=torch.bool)
    subtasks: list[str] = []
    for index, row in enumerate(rows):
        input_ids = encoder.encode_input(row.get("input"), add_eos=False)
        target_ids = encoder.encode_target(row.get("target"), add_eos=False)
        if len(input_ids) != 2048 or len(target_ids) != 1024:
            raise ValueError(f"copy_corrected_v01 row has lengths input={len(input_ids)} target={len(target_ids)}")
        if input_ids[:1024] != target_ids:
            raise ValueError("copy_corrected_v01 row source prefix does not equal target")
        if input_ids[1024:] != [marker_id] * 1024:
            raise ValueError("copy_corrected_v01 row marker suffix is invalid")
        tokens[index] = torch.tensor(input_ids, dtype=torch.long)
        targets[index] = torch.tensor(target_ids, dtype=torch.long)
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        subtasks.append(str(meta.get("ruler_task") or row.get("variant") or "copy_corrected_v01"))
    return ProbeBatch(
        tokens=tokens.to(device),
        target_positions=target_positions.to(device),
        targets=targets.to(device),
        target_mask=target_mask.to(device),
        class_targets=None,
        pad_mask=valid_token_mask.to(device),
        subtasks=subtasks,
        example_count=batch_size,
        token_count=batch_size * 2048,
    )


class ProbeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_output_size: int,
        class_count: int,
        seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
        position_encoding: str = "learned_absolute",
        rope_theta: float = 10000.0,
        use_class_head: bool = True,
    ):
        super().__init__()
        if position_encoding not in {"learned_absolute", "rope"}:
            raise ValueError(f"unknown position_encoding={position_encoding!r}")
        self.token = nn.Embedding(vocab_size, d_model)
        self.position_encoding = position_encoding
        self.pos = nn.Embedding(seq_len, d_model) if position_encoding == "learned_absolute" else None
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    heads,
                    ffn_dim,
                    dropout,
                    attention_backend,
                    block_size,
                    position_encoding=position_encoding,
                    rope_theta=rope_theta,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, token_output_size)
        self.class_head = nn.Linear(d_model, class_count) if use_class_head else None

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        mask: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        valid_neighbors: torch.Tensor | None = None,
        block_pair_index: torch.Tensor | None = None,
        local_log_m: torch.Tensor | None = None,
        neighbor_log_m: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.token(tokens)
        if self.pos is not None:
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            h = h + self.pos(pos)[None, :, :]
        for block in self.blocks:
            h = block(
                h,
                mask,
                local_valid,
                neighbors,
                valid_neighbors,
                block_pair_index,
                local_log_m,
                neighbor_log_m,
            )
        h = self.norm(h)
        token_logits = self.token_head(h)
        if self.class_head is None:
            return token_logits, token_logits.new_empty((tokens.shape[0], 0))
        weights = pad_mask.float()
        pooled = (h * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)[:, None]
        class_logits = self.class_head(pooled)
        return token_logits, class_logits


def gather_position_logits(token_logits: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    batch, _, vocab = token_logits.shape
    expanded = target_positions[:, :, None].expand(batch, target_positions.shape[1], vocab)
    return token_logits.gather(1, expanded)


def parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def padded_length(raw_length: int, block_size: int) -> int:
    return int(math.ceil(int(raw_length) / int(block_size)) * int(block_size))
