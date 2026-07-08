from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.backends import BackendBundle
from src.model.rotary import RotaryEmbedding, apply_rotary_pos_emb


@dataclass(frozen=True)
class MemoryRolloutConfig:
    enabled: bool
    alpha: float
    injection_scale: float
    head_merge: str
    update: str
    initial_state: str


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    output_size: int
    num_layers: int
    dim: int
    dim_ffn: int
    num_heads: int
    activation: str
    dropout: float
    attention_dropout: float
    norm_type: str
    positional_encoding: str
    sequence_length: int
    use_rope: bool
    rope_theta: float
    memory_rollout: MemoryRolloutConfig
    class_count: int | None = None
    class_pooling: str = "mean"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm_type={norm_type!r}")


def make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation={name!r}")


def rollout_memory_update(
    memory_state: torch.Tensor,
    head_transition: torch.Tensor,
    config: MemoryRolloutConfig,
) -> torch.Tensor:
    if config.head_merge != "mean":
        raise ValueError(f"unknown memory_rollout head_merge={config.head_merge!r}")
    if config.update != "lazy":
        raise ValueError(f"unknown memory_rollout update={config.update!r}")
    transition = head_transition.mean(dim=1)
    carried = torch.matmul(transition, memory_state)
    return float(config.alpha) * memory_state + (1.0 - float(config.alpha)) * carried


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.dim) // int(config.num_heads)
        self.norm1 = make_norm(config.norm_type, config.dim)
        self.qkv = nn.Linear(config.dim, config.dim * 3)
        self.out = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = make_norm(config.norm_type, config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim_ffn),
            make_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_ffn, config.dim),
            nn.Dropout(config.dropout),
        )
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta) if config.use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        backend,
        memory_state: torch.Tensor | None,
        memory_config: MemoryRolloutConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        batch, seq_len, dim = h.shape
        qkv = self.qkv(h).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rotary is not None:
            cos, sin = self.rotary(seq_len, q.device, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        transition = None
        if memory_config.enabled:
            transition = backend.transition(q, k)
            attn = torch.matmul(backend.dropout(transition), v)
        else:
            attn = backend(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        x = x + self.dropout(self.out(attn))
        hidden = x + self.ffn(self.norm2(x))
        if memory_config.enabled:
            if memory_state is None:
                raise ValueError("memory_state is required when memory_rollout is enabled")
            if transition is None:
                raise ValueError("transition is required when memory_rollout is enabled")
            memory_state = rollout_memory_update(memory_state, transition, memory_config)
            hidden = hidden + float(memory_config.injection_scale) * memory_state
        return hidden, memory_state


class SequenceTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.positional_encoding not in {"learned_absolute", "rope", "none"}:
            raise ValueError(f"unknown positional_encoding={config.positional_encoding!r}")
        self.token = nn.Embedding(config.vocab_size, config.dim)
        use_learned = config.positional_encoding == "learned_absolute"
        self.position = nn.Embedding(config.sequence_length, config.dim) if use_learned else None
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = make_norm(config.norm_type, config.dim)
        self.token_head = nn.Linear(config.dim, config.output_size)
        self.class_head = nn.Linear(config.dim, config.class_count) if config.class_count is not None else None
        if config.class_pooling not in {"mean", "last"}:
            raise ValueError(f"unknown class_pooling={config.class_pooling!r}")

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        backends: BackendBundle,
        *,
        return_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if len(backends) != len(self.blocks):
            raise ValueError("backend count must equal model layers")
        h = self.token(tokens)
        if self.position is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            h = h + self.position(positions)[None, :, :]
        if self.config.memory_rollout.enabled and self.config.memory_rollout.initial_state != "input":
            raise ValueError(f"unknown memory_rollout initial_state={self.config.memory_rollout.initial_state!r}")
        memory_state = h if self.config.memory_rollout.enabled else None
        for block, backend in zip(self.blocks, backends.backends):
            h, memory_state = block(h, backend, memory_state, self.config.memory_rollout)
        h = self.norm(h)
        token_logits = self.token_head(h)
        class_logits = None
        if self.class_head is not None:
            if self.config.class_pooling == "mean":
                mask = pad_mask.to(dtype=h.dtype).unsqueeze(-1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                last_indices = pad_mask.long().sum(dim=1).clamp_min(1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), last_indices]
            class_logits = self.class_head(pooled)
        if return_memory:
            return token_logits, class_logits, memory_state
        return token_logits, class_logits


def model_config_from_resolved(config: dict) -> ModelConfig:
    model = config["model"]
    task = config["task"]
    rope = model.get("rope", {})
    memory = config["memory_rollout"]
    class_count = int(task["output_size"]) if task.get("loss_type") == "classification" else None
    return ModelConfig(
        vocab_size=int(task["vocab_size"]),
        output_size=int(task["output_size"]),
        num_layers=int(model["num_layers"]),
        dim=int(model["dim"]),
        dim_ffn=int(model["dim_ffn"]),
        num_heads=int(model["num_heads"]),
        activation=str(model["activation"]),
        dropout=float(model["dropout"]),
        attention_dropout=float(model["attention_dropout"]),
        norm_type=str(model["norm_type"]),
        positional_encoding=str(model["positional_encoding"]),
        sequence_length=int(task["sequence_length"]),
        use_rope=bool(rope.get("enabled", False)) or str(model["positional_encoding"]) == "rope",
        rope_theta=float(rope.get("theta", 10000.0)),
        memory_rollout=MemoryRolloutConfig(
            enabled=bool(memory["enabled"]),
            alpha=float(memory["alpha"]),
            injection_scale=float(memory["injection_scale"]),
            head_merge=str(memory["head_merge"]),
            update=str(memory["update"]),
            initial_state=str(memory["initial_state"]),
        ),
        class_count=class_count,
        class_pooling=str(model.get("class_pooling", "mean")),
    )
