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
import torch.nn.functional as F

from synthetic_mvp_core.model import Block, apply_rotary_pos_emb
from synthetic_mvp_core.attention import split_attention_rollout


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
        elif self.encoder_type == "cl100k_base_shift":
            import tiktoken

            text = value if isinstance(value, str) else target_to_text(value)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = [int(token) + 1 for token in encoding.encode(text)]
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
        if self.encoder_type == "cl100k_base_shift":
            return self.encode_input(target_to_text(value), add_eos=False)
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
        if self.encoder_type == "cl100k_base_shift":
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            raw = [max(0, int(value) - 1) for value in values if int(value) > 0]
            return encoding.decode(raw)
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


def cl100k_base_shift_encoder(output_path: Path) -> dict:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    payload = {
        "encoder_type": "cl100k_base_shift",
        "vocab_size": int(encoding.n_vocab) + 2,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": None,
        "unk_token_id": None,
        "integer_shift": 1,
        "readout_token_id": int(encoding.n_vocab) + 1,
        "base_encoding": "cl100k_base",
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
    if bool(task_record.get("no_target_append_v01", False)):
        return make_no_target_append_batch(rows, task_record, encoder, device)
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


def _encoded_target_len(row: dict, encoder: ProbeEncoder) -> int:
    return len(encoder.encode_target(row.get("target"), add_eos=False))


def make_no_target_append_batch(rows: list[dict], task_record: dict, encoder: ProbeEncoder, device: torch.device) -> ProbeBatch:
    """Build batches for the corrected probe contract.

    The old v08 path appended target readout slots after the input.  This
    function forbids that: sequence supervision positions must already lie
    inside the original input tensor, or the task must be classification.
    For RULER-style text rows that do not contain answer slots, the frozen
    corrected contract replaces the final K encoded input tokens with a
    readout sentinel while keeping the encoded input length unchanged.
    """
    task = str(task_record["task"])
    loss_type = str(task_record["resolved_loss_type"])
    T = int(task_record["resolved_padded_sequence_length"])
    input_limit = int(task_record["resolved_runtime_input_length"])
    target_limit = int(task_record["resolved_runtime_target_length"])
    tokens = torch.full((len(rows), T), encoder.pad_token_id, dtype=torch.long)
    pad_mask = torch.zeros((len(rows), T), dtype=torch.bool)
    subtasks: list[str] = []
    target_positions = None
    targets = None
    target_mask = None
    class_targets = None
    token_count = 0

    if loss_type in {"sequence_cross_entropy", "retrieval_sequence_cross_entropy", "mqar_position_cross_entropy"}:
        target_positions = torch.zeros((len(rows), target_limit), dtype=torch.long)
        targets = torch.zeros((len(rows), target_limit), dtype=torch.long)
        target_mask = torch.zeros((len(rows), target_limit), dtype=torch.bool)
    elif loss_type == "classification_cross_entropy":
        class_targets = torch.zeros((len(rows),), dtype=torch.long)
    else:
        raise ValueError(f"unsupported loss_type={loss_type!r}")

    for batch_index, row in enumerate(rows):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        subtasks.append(str(meta.get("ruler_task") or row.get("variant") or "not_applicable"))

        if loss_type == "classification_cross_entropy":
            input_ids = encoder.encode_input(row.get("input"), add_eos=False)[:input_limit]
            if len(input_ids) > T:
                raise ValueError(f"{task} encoded input length {len(input_ids)} exceeds T={T}")
            if input_ids:
                tokens[batch_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
                pad_mask[batch_index, : len(input_ids)] = True
                token_count += len(input_ids)
            assert class_targets is not None
            class_targets[batch_index] = int(row.get("target"))
            continue

        assert target_positions is not None and targets is not None and target_mask is not None

        if task == "selective_copy":
            input_ids = encoder.encode_input(row.get("input"), add_eos=False)
            if len(input_ids) > input_limit:
                raise ValueError(f"selective_copy input length {len(input_ids)} exceeds input_limit={input_limit}")
            target_ids = encoder.encode_target(row.get("target"), add_eos=False)
            readout_len = len(target_ids)
            start = len(input_ids) - readout_len
            if start < 0 or readout_len > target_limit:
                raise ValueError("selective_copy target length does not fit original input")
            if input_ids:
                tokens[batch_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
                pad_mask[batch_index, : len(input_ids)] = True
                token_count += len(input_ids)
            for offset, target_id in enumerate(target_ids):
                pos = start + offset
                if pos >= len(input_ids):
                    raise ValueError("selective_copy readout position escaped original input")
                target_positions[batch_index, offset] = pos
                targets[batch_index, offset] = int(target_id)
                target_mask[batch_index, offset] = True
            continue

        if task == "induction_associative_recall":
            input_ids = encoder.encode_input(row.get("input"), add_eos=False)
            if len(input_ids) > input_limit:
                raise ValueError(f"mqar input length {len(input_ids)} exceeds input_limit={input_limit}")
            if input_ids:
                tokens[batch_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
                pad_mask[batch_index, : len(input_ids)] = True
                token_count += len(input_ids)
            entries = row.get("target", [])
            if len(entries) > target_limit:
                raise ValueError(f"mqar target entries {len(entries)} exceed target_limit={target_limit}")
            for offset, item in enumerate(entries):
                pos = int(item["position"])
                value = encoder.encode_target(int(item["value"]), add_eos=False)[0]
                if not (0 <= pos < len(input_ids)):
                    raise ValueError(f"mqar target position {pos} outside original input length {len(input_ids)}")
                target_positions[batch_index, offset] = pos
                targets[batch_index, offset] = int(value)
                target_mask[batch_index, offset] = True
            continue

        if task in {"niah_kv_retrieval", "ruler"}:
            if str(task_record.get("text_readout_replacement_policy", "blocked")) != "replace_tail_with_readout_sentinel":
                raise ValueError(
                    f"{task} rows keep the answer outside the original input and do not contain "
                    "an in-input answer/readout slot. no_target_append_v01 therefore blocks this "
                    "task unless a manifest explicitly declares a text_readout_replacement_policy."
                )
            input_ids = encoder.encode_input(row.get("input"), add_eos=False)[:input_limit]
            target_ids = encoder.encode_target(row.get("target"), add_eos=False)
            if len(target_ids) > target_limit:
                raise ValueError(f"{task} encoded target length {len(target_ids)} exceeds target_limit={target_limit}")
            if len(input_ids) < target_limit:
                raise ValueError(f"{task} input shorter than reserved readout window")
            if len(input_ids) > T:
                raise ValueError(f"{task} encoded input length {len(input_ids)} exceeds T={T}")
            input_ids = list(input_ids)
            readout_token = int(encoder.payload.get("readout_token_id", encoder.unk_token_id or encoder.pad_token_id))
            readout_start = len(input_ids) - target_limit
            for pos in range(readout_start, len(input_ids)):
                input_ids[pos] = readout_token
            if input_ids:
                tokens[batch_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
                pad_mask[batch_index, : len(input_ids)] = True
                token_count += len(input_ids)
            for offset, target_id in enumerate(target_ids):
                pos = readout_start + offset
                if pos >= len(input_ids):
                    raise ValueError(f"{task} readout position escaped original input")
                target_positions[batch_index, offset] = pos
                targets[batch_index, offset] = int(target_id)
                target_mask[batch_index, offset] = True
            continue

        raise ValueError(f"no_target_append_v01 does not define task={task!r}")

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
        edge_bias_shapes: list[tuple[int, int, int]] | None = None,
        learned_attention_edge_bias: bool = False,
        learned_edge_memory_transport_mode: str | None = None,
        learned_edge_memory_transport_scale: float = 1.0,
        learned_edge_memory_transport_temperature: float = 1.0,
        learned_edge_bias_init: float = 0.0,
        value_position_encoding: str = "none",
        relative_attention_bias: bool = False,
        relative_attention_bias_init: float = 0.0,
        attention_logit_scale_multiplier: float = 1.0,
        learned_attention_logit_scale: bool = False,
        attention_top_k: int = 0,
        rollout_memory: bool = False,
        rollout_memory_scale: float = 1.0,
        rollout_memory_source: str = "input",
        rollout_memory_update: str = "replace",
        rollout_memory_lazy_alpha: float = 0.0,
        rollout_memory_learned_update: bool = False,
        rollout_memory_learned_scale: bool = False,
        rollout_memory_steps: int = 1,
        rollout_memory_multiscale_steps: list[int] | tuple[int, ...] | None = None,
        rollout_memory_multiscale_weights: list[float] | tuple[float, ...] | None = None,
        rollout_head_merge: str = "mean",
        rollout_weight_mode: str = "soft",
        rollout_edge_scope: str = "all",
        rollout_local_logit_bias: float = 0.0,
        rollout_cross_logit_bias: float = 0.0,
        rollout_output_logits: bool = False,
        rollout_output_scale: float = 1.0,
        rollout_output_mode: str = "shared_head",
        attention_residual_scale: float = 1.0,
        ffn_residual_scale: float = 1.0,
        history_output_logits: bool = False,
        history_output_scale: float = 1.0,
        history_output_source: str = "hidden",
        history_output_merge: str = "concat",
        history_include_input: bool = True,
        positional_rollout_memory: bool = False,
        positional_rollout_scale: float = 1.0,
        positional_rollout_update: str = "replace",
        positional_rollout_head_merge: str = "mean",
        positional_rollout_output_logits: bool = False,
        positional_rollout_output_scale: float = 1.0,
        positional_rollout_output_mode: str = "shared_head",
        token_rollout_memory: bool = False,
        token_rollout_scale: float = 1.0,
        token_rollout_logit_mode: str = "prob",
    ):
        super().__init__()
        if position_encoding not in {"learned_absolute", "rope"}:
            raise ValueError(f"unknown position_encoding={position_encoding!r}")
        if value_position_encoding not in {"none", "rope_relative"}:
            raise ValueError(f"unknown value_position_encoding={value_position_encoding!r}")
        if value_position_encoding != "none" and position_encoding != "rope":
            raise ValueError("value_position_encoding requires RoPE position_encoding")
        edge_memory_modes = {None, "none", "residual", "replace", "residual_update"}
        if learned_edge_memory_transport_mode not in edge_memory_modes:
            raise ValueError(
                f"unknown learned_edge_memory_transport_mode={learned_edge_memory_transport_mode!r}"
            )
        self.token = nn.Embedding(vocab_size, d_model)
        self.token_output_size = int(token_output_size)
        self.position_encoding = position_encoding
        self.value_position_encoding = value_position_encoding
        self.relative_attention_bias = bool(relative_attention_bias)
        self.attention_logit_scale_multiplier = float(attention_logit_scale_multiplier)
        self.learned_attention_logit_scale = bool(learned_attention_logit_scale)
        self.attention_top_k = int(attention_top_k)
        if self.attention_top_k < 0:
            raise ValueError("attention_top_k must be non-negative")
        self.rollout_memory = bool(rollout_memory)
        self.rollout_memory_scale = float(rollout_memory_scale)
        self.rollout_memory_source = str(rollout_memory_source)
        if self.rollout_memory_source not in {"input", "hidden"}:
            raise ValueError(f"unknown rollout_memory_source={rollout_memory_source!r}")
        self.rollout_memory_update = str(rollout_memory_update)
        if self.rollout_memory_update not in {"replace", "residual", "lazy"}:
            raise ValueError(f"unknown rollout_memory_update={self.rollout_memory_update!r}")
        self.rollout_memory_lazy_alpha = float(rollout_memory_lazy_alpha)
        if not 0.0 <= self.rollout_memory_lazy_alpha <= 1.0:
            raise ValueError("rollout_memory_lazy_alpha must be in [0, 1]")
        self.rollout_memory_learned_update = bool(rollout_memory_learned_update)
        self.rollout_memory_learned_scale = bool(rollout_memory_learned_scale)
        if self.rollout_memory_learned_update and not self.rollout_memory:
            raise ValueError("rollout_memory_learned_update requires rollout_memory")
        if self.rollout_memory_learned_scale and not self.rollout_memory:
            raise ValueError("rollout_memory_learned_scale requires rollout_memory")
        self.rollout_memory_steps = int(rollout_memory_steps)
        if self.rollout_memory_steps < 1:
            raise ValueError("rollout_memory_steps must be >= 1")
        if rollout_memory_multiscale_steps is None:
            rollout_memory_multiscale_steps = [self.rollout_memory_steps]
        self.rollout_memory_multiscale_steps = [int(step) for step in rollout_memory_multiscale_steps]
        if not self.rollout_memory_multiscale_steps:
            raise ValueError("rollout_memory_multiscale_steps must not be empty")
        if any(step < 1 for step in self.rollout_memory_multiscale_steps):
            raise ValueError("rollout_memory_multiscale_steps values must be >= 1")
        if rollout_memory_multiscale_weights is None:
            self.rollout_memory_multiscale_weights = [
                1.0 / float(len(self.rollout_memory_multiscale_steps))
                for _ in self.rollout_memory_multiscale_steps
            ]
        else:
            self.rollout_memory_multiscale_weights = [float(weight) for weight in rollout_memory_multiscale_weights]
            if len(self.rollout_memory_multiscale_weights) != len(self.rollout_memory_multiscale_steps):
                raise ValueError("rollout_memory_multiscale_weights length must match rollout_memory_multiscale_steps")
            if any(weight < 0.0 for weight in self.rollout_memory_multiscale_weights):
                raise ValueError("rollout_memory_multiscale_weights values must be non-negative")
            weight_sum = sum(self.rollout_memory_multiscale_weights)
            if weight_sum <= 0.0:
                raise ValueError("rollout_memory_multiscale_weights must have positive sum")
            self.rollout_memory_multiscale_weights = [
                weight / float(weight_sum) for weight in self.rollout_memory_multiscale_weights
            ]
        self.rollout_head_merge = str(rollout_head_merge)
        if self.rollout_head_merge not in {"mean", "concat_linear"}:
            raise ValueError(f"unknown rollout_head_merge={self.rollout_head_merge!r}")
        self.rollout_weight_mode = str(rollout_weight_mode)
        if self.rollout_weight_mode not in {"soft", "straight_through_hard"}:
            raise ValueError(f"unknown rollout_weight_mode={self.rollout_weight_mode!r}")
        self.rollout_edge_scope = str(rollout_edge_scope)
        if self.rollout_edge_scope not in {"all", "cross_only", "local_only"}:
            raise ValueError(f"unknown rollout_edge_scope={self.rollout_edge_scope!r}")
        self.rollout_local_logit_bias = float(rollout_local_logit_bias)
        self.rollout_cross_logit_bias = float(rollout_cross_logit_bias)
        self.rollout_output_logits = bool(rollout_output_logits)
        self.rollout_output_scale = float(rollout_output_scale)
        self.rollout_output_mode = str(rollout_output_mode)
        if self.rollout_output_mode not in {"shared_head", "separate_head", "embedding_tied", "embedding_tied_cosine"}:
            raise ValueError(f"unknown rollout_output_mode={self.rollout_output_mode!r}")
        if self.rollout_output_logits and not self.rollout_memory:
            raise ValueError("rollout_output_logits requires rollout_memory")
        self.attention_residual_scale = float(attention_residual_scale)
        self.ffn_residual_scale = float(ffn_residual_scale)
        self.history_output_logits = bool(history_output_logits)
        self.history_output_scale = float(history_output_scale)
        self.history_output_source = str(history_output_source)
        if self.history_output_source not in {"hidden", "rollout", "hidden_rollout"}:
            raise ValueError(f"unknown history_output_source={self.history_output_source!r}")
        self.history_output_merge = str(history_output_merge)
        if self.history_output_merge not in {
            "concat",
            "weighted_sum",
            "logit_weighted_sum",
            "confidence_logit_weighted_sum",
        }:
            raise ValueError(f"unknown history_output_merge={self.history_output_merge!r}")
        self.history_include_input = bool(history_include_input)
        if (
            self.history_output_logits
            and self.history_output_source in {"rollout", "hidden_rollout"}
            and not self.rollout_memory
        ):
            raise ValueError("rollout history output requires rollout_memory")
        self.positional_rollout_memory = bool(positional_rollout_memory)
        self.positional_rollout_scale = float(positional_rollout_scale)
        self.positional_rollout_update = str(positional_rollout_update)
        if self.positional_rollout_update not in {"replace", "residual"}:
            raise ValueError(f"unknown positional_rollout_update={self.positional_rollout_update!r}")
        self.positional_rollout_head_merge = str(positional_rollout_head_merge)
        if self.positional_rollout_head_merge not in {"mean", "concat_linear"}:
            raise ValueError(f"unknown positional_rollout_head_merge={self.positional_rollout_head_merge!r}")
        self.positional_rollout_output_logits = bool(positional_rollout_output_logits)
        self.positional_rollout_output_scale = float(positional_rollout_output_scale)
        self.positional_rollout_output_mode = str(positional_rollout_output_mode)
        if self.positional_rollout_output_mode not in {
            "shared_head",
            "separate_head",
            "embedding_tied_cosine",
        }:
            raise ValueError(
                f"unknown positional_rollout_output_mode={self.positional_rollout_output_mode!r}"
            )
        if self.positional_rollout_output_logits and not self.positional_rollout_memory:
            raise ValueError("positional_rollout_output_logits requires positional_rollout_memory")
        self.token_rollout_memory = bool(token_rollout_memory)
        self.token_rollout_scale = float(token_rollout_scale)
        self.token_rollout_logit_mode = str(token_rollout_logit_mode)
        if self.token_rollout_logit_mode not in {"prob", "log"}:
            raise ValueError(f"unknown token_rollout_logit_mode={token_rollout_logit_mode!r}")
        self.pos = nn.Embedding(seq_len, d_model) if position_encoding == "learned_absolute" else None
        self.block_size = int(block_size)
        self.learned_attention_edge_bias = bool(learned_attention_edge_bias)
        self.learned_edge_memory_transport_mode = (
            None
            if learned_edge_memory_transport_mode in {None, "none"}
            else str(learned_edge_memory_transport_mode)
        )
        self.learned_edge_memory_transport_scale = float(learned_edge_memory_transport_scale)
        self.learned_edge_memory_transport_temperature = float(learned_edge_memory_transport_temperature)
        if self.learned_edge_memory_transport_temperature <= 0:
            raise ValueError("learned_edge_memory_transport_temperature must be positive")
        use_edge_params = self.learned_attention_edge_bias or self.learned_edge_memory_transport_mode is not None
        if use_edge_params:
            if edge_bias_shapes is None:
                raise ValueError("edge_bias_shapes are required for learned edge parameters")
            if len(edge_bias_shapes) != int(layers):
                raise ValueError(
                    f"edge_bias_shapes length {len(edge_bias_shapes)} does not match layers={layers}"
                )
            self.learned_local_edge_log_bias = nn.ParameterList()
            self.learned_neighbor_edge_log_bias = nn.ParameterList()
            for shape in edge_bias_shapes:
                layer_seq_len, local_width, neighbor_width = (int(shape[0]), int(shape[1]), int(shape[2]))
                if layer_seq_len != int(seq_len):
                    raise ValueError(
                        f"edge bias seq_len {layer_seq_len} does not match model seq_len={seq_len}"
                    )
                self.learned_local_edge_log_bias.append(
                    nn.Parameter(torch.full((layer_seq_len, local_width), float(learned_edge_bias_init)))
                )
                self.learned_neighbor_edge_log_bias.append(
                    nn.Parameter(torch.full((layer_seq_len, neighbor_width), float(learned_edge_bias_init)))
                )
        else:
            self.learned_local_edge_log_bias = None
            self.learned_neighbor_edge_log_bias = None
        if self.relative_attention_bias:
            self.relative_attention_log_bias = nn.Parameter(
                torch.full((int(layers), int(heads), 2 * int(seq_len) - 1), float(relative_attention_bias_init))
            )
        else:
            self.relative_attention_log_bias = None
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
                    value_position_encoding=value_position_encoding,
                    attention_logit_scale_multiplier=attention_logit_scale_multiplier,
                    learned_attention_logit_scale=learned_attention_logit_scale,
                    attention_top_k=self.attention_top_k,
                    attention_residual_scale=self.attention_residual_scale,
                    ffn_residual_scale=self.ffn_residual_scale,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, token_output_size)
        self.rollout_token_head = (
            nn.Linear(d_model, token_output_size)
            if self.rollout_output_logits and self.rollout_output_mode == "separate_head"
            else None
        )
        self.rollout_head_merge_proj = (
            nn.Linear(int(heads) * int(d_model), d_model)
            if self.rollout_memory and self.rollout_head_merge == "concat_linear"
            else None
        )
        self.rollout_memory_update_logits = (
            nn.Parameter(torch.full((int(layers),), self._logit(self.rollout_memory_lazy_alpha)))
            if self.rollout_memory and self.rollout_memory_learned_update
            else None
        )
        self.rollout_memory_scale_logits = (
            nn.Parameter(
                torch.full(
                    (int(layers),),
                    math.log(max(math.exp(float(self.rollout_memory_scale)) - 1.0, 1e-6)),
                )
            )
            if self.rollout_memory and self.rollout_memory_learned_scale
            else None
        )
        history_state_count = int(layers) + (1 if self.history_include_input else 0)
        history_source_multiplier = 2 if self.history_output_source == "hidden_rollout" else 1
        history_part_count = history_state_count * history_source_multiplier
        history_width = history_part_count * int(d_model) if self.history_output_merge == "concat" else int(d_model)
        self.history_norm = nn.LayerNorm(history_width) if self.history_output_logits else None
        self.history_token_head = nn.Linear(history_width, token_output_size) if self.history_output_logits else None
        self.history_mix_logits = (
            nn.Parameter(torch.zeros(history_part_count))
            if self.history_output_logits
            and self.history_output_merge in {"weighted_sum", "logit_weighted_sum", "confidence_logit_weighted_sum"}
            else None
        )
        self.positional_rollout_head_merge_proj = (
            nn.Linear(int(heads) * int(d_model), d_model)
            if self.positional_rollout_memory and self.positional_rollout_head_merge == "concat_linear"
            else None
        )
        self.positional_rollout_token_head = (
            nn.Linear(d_model, token_output_size)
            if self.positional_rollout_output_logits
            and self.positional_rollout_output_mode == "separate_head"
            else None
        )
        self.class_head = nn.Linear(d_model, class_count) if use_class_head else None

    @staticmethod
    def _logit(value: float) -> float:
        clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(clipped / (1.0 - clipped))

    @staticmethod
    def _add_optional_bias(base: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return base
        if base is None:
            return bias
        return base + bias

    def _learned_edge_biases(self, layer_index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.learned_local_edge_log_bias is None or self.learned_neighbor_edge_log_bias is None:
            return None, None
        return self.learned_local_edge_log_bias[layer_index], self.learned_neighbor_edge_log_bias[layer_index]

    def _relative_edge_biases(
        self,
        layer_index: int,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.relative_attention_log_bias is None:
            return None, None
        seq_len = int(local_valid.shape[0])
        block_size = int(self.block_size)
        table = self.relative_attention_log_bias[layer_index]
        query_positions = torch.arange(seq_len, device=local_valid.device)
        offsets = torch.arange(block_size, device=local_valid.device)
        block_starts = (query_positions // block_size) * block_size
        local_positions = block_starts[:, None] + offsets[None, :]
        local_index = local_positions - query_positions[:, None] + (seq_len - 1)
        local_bias = table[:, local_index]
        neighbor_bias = None
        if neighbors is not None and int(neighbors.shape[1]) > 0:
            neighbor_index = neighbors.to(device=local_valid.device) - query_positions[:, None] + (seq_len - 1)
            neighbor_bias = table[:, neighbor_index]
        return local_bias, neighbor_bias

    def _edge_memory_transport(
        self,
        route_state: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        local_edge_log_bias: torch.Tensor | None,
        neighbor_edge_log_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if local_edge_log_bias is None:
            raise ValueError("learned edge memory transport requires local edge logits")
        batch, seq_len, d_model = route_state.shape
        block_size = int(self.block_size)
        if seq_len % block_size != 0:
            raise ValueError("seq_len must be divisible by block_size for edge memory transport")
        offsets = torch.arange(block_size, device=route_state.device)
        block_starts = (torch.arange(seq_len, device=route_state.device) // block_size) * block_size
        local_positions = block_starts[:, None] + offsets[None, :]
        local_values = route_state[:, local_positions.reshape(-1), :].reshape(
            batch, seq_len, block_size, d_model
        )
        temperature = float(self.learned_edge_memory_transport_temperature)
        local_scores = local_edge_log_bias.to(device=route_state.device, dtype=route_state.dtype) / temperature
        local_scores = local_scores.masked_fill(
            ~local_valid.to(device=route_state.device),
            torch.finfo(route_state.dtype).min,
        )

        has_cross = (
            neighbors is not None
            and valid_neighbors is not None
            and neighbor_edge_log_bias is not None
            and neighbors.shape[1] > 0
        )
        if has_cross:
            cross_values = route_state[:, neighbors.reshape(-1), :].reshape(
                batch, seq_len, neighbors.shape[1], d_model
            )
            cross_scores = neighbor_edge_log_bias.to(device=route_state.device, dtype=route_state.dtype) / temperature
            cross_scores = cross_scores.masked_fill(
                ~valid_neighbors.to(device=route_state.device),
                torch.finfo(route_state.dtype).min,
            )
            scores = torch.cat([local_scores, cross_scores], dim=-1)
        else:
            cross_values = None
            scores = local_scores

        weights = torch.softmax(scores, dim=-1)
        local_weights = weights[:, :block_size]
        transported = (local_weights[None, :, :, None] * local_values).sum(dim=2)
        if has_cross and cross_values is not None:
            cross_weights = weights[:, block_size:]
            transported = transported + (cross_weights[None, :, :, None] * cross_values).sum(dim=2)
        return transported

    def _attention_rollout_memory(
        self,
        block: Block,
        h: torch.Tensor,
        memory_state: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        local_log_m: torch.Tensor | None,
        neighbor_log_m: torch.Tensor | None,
    ) -> torch.Tensor:
        if block.attn.attention_backend != "split":
            raise ValueError("rollout_memory currently requires split attention backend")
        x = block.ln1(h)
        batch, seq_len, d_model = x.shape
        qkv = block.attn.qkv(x).view(batch, seq_len, 3, block.attn.num_heads, block.attn.head_dim)
        q, k, _v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if block.attn.rotary_emb is not None:
            cos, sin = block.attn.rotary_emb(seq_len, q.device, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if block.attn.log_attention_logit_scale is not None:
            q = q * block.attn.log_attention_logit_scale.exp().to(device=q.device, dtype=q.dtype)[None, :, None, None]
        elif block.attn.attention_logit_scale_multiplier != 1.0:
            q = q * float(block.attn.attention_logit_scale_multiplier)
        per_head_memory = None
        for rollout_steps, rollout_weight in zip(
            self.rollout_memory_multiscale_steps,
            self.rollout_memory_multiscale_weights,
        ):
            scale_memory = split_attention_rollout(
                q,
                k,
                memory_state,
                self.block_size,
                local_valid,
                neighbors,
                valid_neighbors,
                local_log_m=local_log_m,
                cross_log_m=neighbor_log_m,
                top_k=block.attn.attention_top_k,
                return_heads=self.rollout_head_merge == "concat_linear",
                weight_mode=self.rollout_weight_mode,
                edge_scope=self.rollout_edge_scope,
                local_extra_log_bias=self.rollout_local_logit_bias,
                cross_extra_log_bias=self.rollout_cross_logit_bias,
                rollout_steps=rollout_steps,
            )
            scale_memory = scale_memory * float(rollout_weight)
            per_head_memory = scale_memory if per_head_memory is None else per_head_memory + scale_memory
        if per_head_memory is None:
            raise ValueError("rollout_memory_multiscale_steps must not be empty")
        if self.rollout_head_merge == "concat_linear":
            if self.rollout_head_merge_proj is None:
                raise ValueError("concat rollout head merge projection is not initialized")
            batch, heads, seq_len, dim = per_head_memory.shape
            return self.rollout_head_merge_proj(
                per_head_memory.transpose(1, 2).contiguous().view(batch, seq_len, heads * dim)
            )
        return per_head_memory

    def _positional_rollout_memory(
        self,
        memory_state: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        local_log_m: torch.Tensor | None,
        neighbor_log_m: torch.Tensor | None,
        heads: int,
    ) -> torch.Tensor:
        batch, seq_len, d_model = memory_state.shape
        device = memory_state.device
        dtype = memory_state.dtype
        block_size = int(self.block_size)
        num_blocks = seq_len // block_size
        local_scores = torch.zeros((heads, seq_len, block_size), device=device, dtype=dtype)
        if local_log_m is not None:
            local_scores = self._add_optional_bias(local_scores, local_log_m)
        local_scores = local_scores.masked_fill(
            ~local_valid.to(device=device)[None, :, :],
            torch.finfo(dtype).min,
        )
        has_cross = (
            neighbors is not None
            and valid_neighbors is not None
            and int(neighbors.shape[1]) > 0
        )
        if has_cross:
            cross_scores = torch.zeros((heads, seq_len, int(neighbors.shape[1])), device=device, dtype=dtype)
            if neighbor_log_m is not None:
                cross_scores = self._add_optional_bias(cross_scores, neighbor_log_m)
            cross_scores = cross_scores.masked_fill(
                ~valid_neighbors.to(device=device)[None, :, :],
                torch.finfo(dtype).min,
            )
            scores = torch.cat([local_scores, cross_scores], dim=-1)
        else:
            scores = local_scores
        weights = torch.softmax(scores, dim=-1)
        memory_blocks = memory_state.view(batch, num_blocks, block_size, d_model)
        local_weights = weights[..., :block_size].reshape(heads, num_blocks, block_size, block_size)
        local_out = torch.einsum("hqts,bqsd->bhqtd", local_weights, memory_blocks).reshape(
            batch, heads, seq_len, d_model
        )
        if has_cross:
            gathered_memory = memory_state[:, neighbors.reshape(-1), :].reshape(
                batch, seq_len, int(neighbors.shape[1]), d_model
            )
            cross_weights = weights[..., block_size:]
            local_out = local_out + torch.stack(
                [
                    (cross_weights[head_index][None, :, :, None] * gathered_memory).sum(dim=-2)
                    for head_index in range(heads)
                ],
                dim=1,
            )
        if self.positional_rollout_head_merge == "concat_linear":
            if self.positional_rollout_head_merge_proj is None:
                raise ValueError("concat positional rollout projection is not initialized")
            return self.positional_rollout_head_merge_proj(
                local_out.transpose(1, 2).contiguous().view(batch, seq_len, heads * d_model)
            )
        return local_out.mean(dim=1)

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
        route_transport_src: torch.Tensor | None = None,
        route_transport_dst: torch.Tensor | None = None,
        route_transport_scale: float | None = None,
        route_transport_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.token(tokens)
        if self.pos is not None:
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            h = h + self.pos(pos)[None, :, :]
        def layer_value(value, layer_index: int):
            if isinstance(value, (list, tuple)):
                return value[layer_index]
            return value

        route_state = h
        rollout_state = h
        positional_rollout_state = h
        hidden_history = [h] if self.history_output_logits and self.history_include_input else []
        rollout_history = (
            [rollout_state]
            if self.history_output_logits
            and self.history_include_input
            and self.history_output_source in {"rollout", "hidden_rollout"}
            else []
        )
        token_rollout_state = (
            F.one_hot(tokens.clamp_min(0), num_classes=self.token_output_size).to(dtype=h.dtype)
            if self.token_rollout_memory
            else None
        )
        for layer_index, block in enumerate(self.blocks):
            layer_local_log_m = layer_value(local_log_m, layer_index)
            layer_neighbor_log_m = layer_value(neighbor_log_m, layer_index)
            learned_local_bias, learned_neighbor_bias = self._learned_edge_biases(layer_index)
            if self.learned_attention_edge_bias:
                layer_local_log_m = self._add_optional_bias(layer_local_log_m, learned_local_bias)
                layer_neighbor_log_m = self._add_optional_bias(layer_neighbor_log_m, learned_neighbor_bias)
            layer_local_valid = layer_value(local_valid, layer_index)
            layer_neighbors = layer_value(neighbors, layer_index)
            layer_valid_neighbors = layer_value(valid_neighbors, layer_index)
            relative_local_bias, relative_neighbor_bias = self._relative_edge_biases(
                layer_index,
                layer_local_valid,
                layer_neighbors,
            )
            layer_local_log_m = self._add_optional_bias(layer_local_log_m, relative_local_bias)
            layer_neighbor_log_m = self._add_optional_bias(layer_neighbor_log_m, relative_neighbor_bias)
            pre_block_h = h
            h = block(
                h,
                layer_value(mask, layer_index),
                layer_local_valid,
                layer_neighbors,
                layer_valid_neighbors,
                layer_value(block_pair_index, layer_index),
                layer_local_log_m,
                layer_neighbor_log_m,
            )
            if self.rollout_memory:
                memory_source = rollout_state if self.rollout_memory_source == "input" else h
                next_rollout_state = self._attention_rollout_memory(
                    block,
                    pre_block_h,
                    memory_source,
                    layer_local_valid,
                    layer_neighbors,
                    layer_valid_neighbors,
                    layer_local_log_m,
                    layer_neighbor_log_m,
                )
                if self.rollout_memory_update == "residual":
                    rollout_state = rollout_state + next_rollout_state
                elif self.rollout_memory_update == "lazy":
                    if self.rollout_memory_update_logits is not None:
                        lazy_alpha = torch.sigmoid(self.rollout_memory_update_logits[layer_index]).to(
                            device=h.device,
                            dtype=h.dtype,
                        )
                    else:
                        lazy_alpha = torch.tensor(
                            float(self.rollout_memory_lazy_alpha),
                            device=h.device,
                            dtype=h.dtype,
                        )
                    rollout_state = lazy_alpha * rollout_state + (1.0 - lazy_alpha) * next_rollout_state
                else:
                    rollout_state = next_rollout_state
                if self.rollout_memory_scale_logits is not None:
                    rollout_scale = F.softplus(self.rollout_memory_scale_logits[layer_index]).to(
                        device=h.device,
                        dtype=h.dtype,
                    )
                else:
                    rollout_scale = torch.tensor(float(self.rollout_memory_scale), device=h.device, dtype=h.dtype)
                h = h + rollout_scale * rollout_state
            if token_rollout_state is not None:
                token_rollout_state = self._attention_rollout_memory(
                    block,
                    pre_block_h,
                    token_rollout_state,
                    layer_local_valid,
                    layer_neighbors,
                    layer_valid_neighbors,
                    layer_local_log_m,
                    layer_neighbor_log_m,
                )
            if self.positional_rollout_memory:
                next_positional_rollout_state = self._positional_rollout_memory(
                    positional_rollout_state,
                    layer_local_valid,
                    layer_neighbors,
                    layer_valid_neighbors,
                    layer_local_log_m,
                    layer_neighbor_log_m,
                    block.attn.num_heads,
                )
                if self.positional_rollout_update == "residual":
                    positional_rollout_state = positional_rollout_state + next_positional_rollout_state
                else:
                    positional_rollout_state = next_positional_rollout_state
                h = h + float(self.positional_rollout_scale) * positional_rollout_state
            if self.learned_edge_memory_transport_mode is not None:
                transported = self._edge_memory_transport(
                    route_state,
                    layer_local_valid,
                    layer_neighbors,
                    layer_valid_neighbors,
                    learned_local_bias,
                    learned_neighbor_bias,
                )
                if self.learned_edge_memory_transport_mode == "residual":
                    route_state = route_state + transported
                    h = h + float(self.learned_edge_memory_transport_scale) * transported
                elif self.learned_edge_memory_transport_mode == "residual_update":
                    route_state = route_state + transported
                    h = h + float(self.learned_edge_memory_transport_scale) * route_state
                elif self.learned_edge_memory_transport_mode == "replace":
                    route_state = transported
                    h = float(self.learned_edge_memory_transport_scale) * route_state
                else:
                    raise ValueError(
                        f"unknown learned_edge_memory_transport_mode={self.learned_edge_memory_transport_mode!r}"
                    )
            layer_route_src = layer_value(route_transport_src, layer_index)
            layer_route_dst = layer_value(route_transport_dst, layer_index)
            layer_route_scale = layer_value(route_transport_scale, layer_index)
            layer_route_mode = layer_value(route_transport_mode, layer_index)
            if layer_route_src is not None and layer_route_dst is not None:
                if layer_route_scale is None:
                    layer_route_scale = 1.0
                if layer_route_mode is None:
                    layer_route_mode = "residual"
                if layer_route_mode == "residual":
                    transported = h.index_select(1, layer_route_dst)
                    h = h.clone()
                    h[:, layer_route_src, :] = h[:, layer_route_src, :] + float(layer_route_scale) * transported
                    route_state = h
                elif layer_route_mode == "replace":
                    transported = h.index_select(1, layer_route_dst)
                    h = h.clone()
                    h[:, layer_route_src, :] = float(layer_route_scale) * transported
                    route_state = h
                elif layer_route_mode == "memory_residual":
                    transported = route_state.index_select(1, layer_route_dst)
                    route_state = route_state.clone()
                    route_state[:, layer_route_src, :] = (
                        route_state[:, layer_route_src, :] + float(layer_route_scale) * transported
                    )
                    h = h.clone()
                    h[:, layer_route_src, :] = h[:, layer_route_src, :] + route_state[:, layer_route_src, :]
                elif layer_route_mode == "memory_replace":
                    transported = route_state.index_select(1, layer_route_dst)
                    route_state = route_state.clone()
                    route_state[:, layer_route_src, :] = float(layer_route_scale) * transported
                    h = h.clone()
                    h[:, layer_route_src, :] = route_state[:, layer_route_src, :]
                else:
                    raise ValueError(f"unknown route_transport_mode={layer_route_mode!r}")
            if self.history_output_logits:
                hidden_history.append(h)
                if self.history_output_source in {"rollout", "hidden_rollout"}:
                    rollout_history.append(rollout_state)
        h = self.norm(h)
        token_logits = self.token_head(h)
        if self.history_output_logits:
            if self.history_token_head is None or self.history_norm is None:
                raise ValueError("history output head is not initialized")
            history_parts: list[torch.Tensor] = []
            if self.history_output_source in {"hidden", "hidden_rollout"}:
                history_parts.extend(hidden_history)
            if self.history_output_source in {"rollout", "hidden_rollout"}:
                history_parts.extend(rollout_history)
            if not history_parts:
                raise ValueError("history output has no states to read")
            if self.history_output_merge in {"logit_weighted_sum", "confidence_logit_weighted_sum"}:
                if self.history_mix_logits is None:
                    if self.history_output_merge == "logit_weighted_sum":
                        raise ValueError("logit-weighted history output weights are not initialized")
                elif len(history_parts) != int(self.history_mix_logits.numel()):
                    raise ValueError(
                        f"history part count {len(history_parts)} does not match "
                        f"initialized weights {int(self.history_mix_logits.numel())}"
                    )
                part_logits = torch.stack(
                    [
                        self.history_token_head(self.history_norm(part))
                        for part in history_parts
                    ],
                    dim=0,
                )
                if self.history_output_merge == "confidence_logit_weighted_sum":
                    history_weights = torch.softmax(part_logits.max(dim=-1).values, dim=0)
                    mixed_part_logits = (history_weights[..., None] * part_logits).sum(dim=0)
                else:
                    history_weights = torch.softmax(self.history_mix_logits.to(device=h.device, dtype=h.dtype), dim=0)
                    mixed_part_logits = (history_weights[:, None, None, None] * part_logits).sum(dim=0)
                token_logits = token_logits + float(self.history_output_scale) * mixed_part_logits
            elif self.history_output_merge == "weighted_sum":
                if self.history_mix_logits is None:
                    raise ValueError("weighted-sum history output weights are not initialized")
                if len(history_parts) != int(self.history_mix_logits.numel()):
                    raise ValueError(
                        f"history part count {len(history_parts)} does not match "
                        f"initialized weights {int(self.history_mix_logits.numel())}"
                    )
                history_weights = torch.softmax(self.history_mix_logits.to(device=h.device, dtype=h.dtype), dim=0)
                history_stack = torch.stack(history_parts, dim=0)
                history_state = (history_weights[:, None, None, None] * history_stack).sum(dim=0)
            else:
                history_state = torch.cat(history_parts, dim=-1)
            if self.history_output_merge not in {"logit_weighted_sum", "confidence_logit_weighted_sum"}:
                token_logits = token_logits + float(self.history_output_scale) * self.history_token_head(
                    self.history_norm(history_state)
                )
        if self.rollout_output_logits:
            rollout_hidden = self.norm(rollout_state)
            if self.rollout_output_mode == "shared_head":
                rollout_logits = self.token_head(rollout_hidden)
            elif self.rollout_output_mode == "separate_head":
                if self.rollout_token_head is None:
                    raise ValueError("separate rollout output head is not initialized")
                rollout_logits = self.rollout_token_head(rollout_hidden)
            elif self.rollout_output_mode == "embedding_tied":
                if self.token_output_size > self.token.weight.shape[0]:
                    raise ValueError("embedding_tied rollout output requires token_output_size <= vocab_size")
                rollout_logits = F.linear(rollout_hidden, self.token.weight[: self.token_output_size])
            elif self.rollout_output_mode == "embedding_tied_cosine":
                if self.token_output_size > self.token.weight.shape[0]:
                    raise ValueError("embedding_tied_cosine rollout output requires token_output_size <= vocab_size")
                rollout_logits = F.linear(
                    F.normalize(rollout_hidden, dim=-1),
                    F.normalize(self.token.weight[: self.token_output_size], dim=-1),
                )
            else:
                raise ValueError(f"unknown rollout_output_mode={self.rollout_output_mode!r}")
            token_logits = token_logits + float(self.rollout_output_scale) * rollout_logits
        if self.positional_rollout_output_logits:
            positional_hidden = self.norm(positional_rollout_state)
            if self.positional_rollout_output_mode == "shared_head":
                positional_logits = self.token_head(positional_hidden)
            elif self.positional_rollout_output_mode == "separate_head":
                if self.positional_rollout_token_head is None:
                    raise ValueError("separate positional rollout output head is not initialized")
                positional_logits = self.positional_rollout_token_head(positional_hidden)
            elif self.positional_rollout_output_mode == "embedding_tied_cosine":
                if self.token_output_size > self.token.weight.shape[0]:
                    raise ValueError(
                        "embedding_tied_cosine positional rollout output requires token_output_size <= vocab_size"
                    )
                positional_logits = F.linear(
                    F.normalize(positional_hidden, dim=-1),
                    F.normalize(self.token.weight[: self.token_output_size], dim=-1),
                )
            else:
                raise ValueError(
                    f"unknown positional_rollout_output_mode={self.positional_rollout_output_mode!r}"
                )
            token_logits = token_logits + float(self.positional_rollout_output_scale) * positional_logits
        if token_rollout_state is not None:
            if self.token_rollout_logit_mode == "log":
                token_memory_logits = torch.log(token_rollout_state.clamp_min(1.0e-6))
            else:
                token_memory_logits = token_rollout_state
            token_logits = token_logits + float(self.token_rollout_scale) * token_memory_logits
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
