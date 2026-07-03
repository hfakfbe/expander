from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.attention import apply_memory_routes
from src.model.backends import BackendBundle
from src.model.rotary import RotaryEmbedding, apply_rotary_pos_emb


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
    class_count: int | None = None


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

    def forward(self, x: torch.Tensor, backend) -> torch.Tensor:
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
        attn = backend(q, k, v).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        x = x + self.dropout(self.out(attn))
        x = apply_memory_routes(x, backend.memory_routes)
        x = x + self.ffn(self.norm2(x))
        return x


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

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor, backends: BackendBundle) -> tuple[torch.Tensor, torch.Tensor | None]:
        if len(backends) != len(self.blocks):
            raise ValueError("backend count must equal model layers")
        h = self.token(tokens)
        if self.position is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            h = h + self.position(positions)[None, :, :]
        for block, backend in zip(self.blocks, backends.backends):
            h = block(h, backend)
        h = self.norm(h)
        token_logits = self.token_head(h)
        class_logits = None
        if self.class_head is not None:
            mask = pad_mask.to(dtype=h.dtype).unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            class_logits = self.class_head(pooled)
        return token_logits, class_logits


def model_config_from_resolved(config: dict) -> ModelConfig:
    model = config["model"]
    task = config["task"]
    rope = model.get("rope", {})
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
        class_count=class_count,
    )

