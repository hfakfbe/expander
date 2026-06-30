from __future__ import annotations

import torch
import torch.nn as nn
import math

from .attention import local_blockpair_attention, local_cross_attention


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if int(head_dim) % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (float(theta) ** (torch.arange(0, int(head_dim), 2, dtype=torch.float32) / int(head_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(int(seq_len), device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
        position_encoding: str = "learned_absolute",
        rope_theta: float = 10000.0,
        value_position_encoding: str = "none",
        attention_logit_scale_multiplier: float = 1.0,
        learned_attention_logit_scale: bool = False,
        attention_top_k: int = 0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if value_position_encoding not in {"none", "rope_relative"}:
            raise ValueError(f"unknown value_position_encoding={value_position_encoding!r}")
        if value_position_encoding != "none" and position_encoding != "rope":
            raise ValueError("value_position_encoding requires RoPE position_encoding")
        if float(attention_logit_scale_multiplier) <= 0:
            raise ValueError("attention_logit_scale_multiplier must be positive")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend
        self.block_size = block_size
        self.position_encoding = position_encoding
        self.value_position_encoding = value_position_encoding
        self.attention_logit_scale_multiplier = float(attention_logit_scale_multiplier)
        self.attention_top_k = int(attention_top_k)
        if self.attention_top_k < 0:
            raise ValueError("attention_top_k must be non-negative")
        if learned_attention_logit_scale:
            init = math.log(float(attention_logit_scale_multiplier))
            self.log_attention_logit_scale = nn.Parameter(torch.full((int(num_heads),), init))
        else:
            self.log_attention_logit_scale = None
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, theta=rope_theta) if position_encoding == "rope" else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        block_pair_index: torch.Tensor | None,
        local_log_m: torch.Tensor | None,
        neighbor_log_m: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        value_rope_cos = None
        value_rope_sin = None
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(seq_len, q.device, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.value_position_encoding == "rope_relative":
                value_rope_cos = cos
                value_rope_sin = sin
                v = (v * cos) + (rotate_half(v) * sin)
        elif self.value_position_encoding != "none":
            raise ValueError("value_position_encoding requires RoPE position_encoding")
        if self.log_attention_logit_scale is not None:
            q = q * self.log_attention_logit_scale.exp().to(device=q.device, dtype=q.dtype)[None, :, None, None]
        elif self.attention_logit_scale_multiplier != 1.0:
            q = q * float(self.attention_logit_scale_multiplier)
        if self.attention_backend == "dense_mask":
            scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)
        elif self.attention_backend == "neighbor":
            if neighbors is None or valid_neighbors is None:
                raise ValueError("neighbor backend requires precomputed neighbor tables")
            batch, heads, seq_len, head_dim = q.shape
            gathered_k = k[:, :, neighbors.reshape(-1), :].reshape(
                batch, heads, seq_len, neighbors.shape[1], head_dim
            )
            gathered_v = v[:, :, neighbors.reshape(-1), :].reshape(
                batch, heads, seq_len, neighbors.shape[1], head_dim
            )
            scores = (q[:, :, :, None, :] * gathered_k).sum(dim=-1) * (head_dim ** -0.5)
            if neighbor_log_m is not None:
                scores = scores + neighbor_log_m[None, None, :, :]
            scores = scores.masked_fill(
                ~valid_neighbors[None, None, :, :], torch.finfo(scores.dtype).min
            )
            if self.attention_top_k > 0 and self.attention_top_k < int(scores.shape[-1]):
                values, _indices = torch.topk(scores, k=self.attention_top_k, dim=-1)
                threshold = values[..., -1, None]
                scores = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            out = (attn[..., None] * gathered_v).sum(dim=-2)
        elif self.attention_backend == "split":
            out = local_cross_attention(
                q,
                k,
                v,
                self.block_size,
                local_valid,
                neighbors,
                valid_neighbors,
                local_log_m=local_log_m,
                cross_log_m=neighbor_log_m,
                dropout=self.dropout,
                top_k=self.attention_top_k,
            )
        elif self.attention_backend == "blockpair":
            out = local_blockpair_attention(
                q,
                k,
                v,
                self.block_size,
                local_valid,
                neighbors,
                valid_neighbors,
                block_pair_index,
                local_log_m=local_log_m,
                cross_log_m=neighbor_log_m,
                dropout=self.dropout,
                top_k=self.attention_top_k,
            )
        else:
            raise ValueError(f"unknown attention backend: {self.attention_backend}")
        if value_rope_cos is not None and value_rope_sin is not None:
            out = (out * value_rope_cos) - (rotate_half(out) * value_rope_sin)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(out)

class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
        position_encoding: str = "learned_absolute",
        rope_theta: float = 10000.0,
        value_position_encoding: str = "none",
        attention_logit_scale_multiplier: float = 1.0,
        learned_attention_logit_scale: bool = False,
        attention_top_k: int = 0,
        attention_residual_scale: float = 1.0,
        ffn_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.attention_residual_scale = float(attention_residual_scale)
        self.ffn_residual_scale = float(ffn_residual_scale)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedSelfAttention(
            d_model,
            num_heads,
            dropout,
            attention_backend,
            block_size,
            position_encoding=position_encoding,
            rope_theta=rope_theta,
            value_position_encoding=value_position_encoding,
            attention_logit_scale_multiplier=attention_logit_scale_multiplier,
            learned_attention_logit_scale=learned_attention_logit_scale,
            attention_top_k=attention_top_k,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None,
        valid_neighbors: torch.Tensor | None,
        block_pair_index: torch.Tensor | None,
        local_log_m: torch.Tensor | None,
        neighbor_log_m: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + float(self.attention_residual_scale) * self.attn(
            self.ln1(x),
            mask,
            local_valid,
            neighbors,
            valid_neighbors,
            block_pair_index,
            local_log_m,
            neighbor_log_m,
        )
        return x + float(self.ffn_residual_scale) * self.ffn(self.ln2(x))

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
        position_encoding: str = "learned_absolute",
        rope_theta: float = 10000.0,
        value_position_encoding: str = "none",
        attention_logit_scale_multiplier: float = 1.0,
        learned_attention_logit_scale: bool = False,
        attention_top_k: int = 0,
        attention_residual_scale: float = 1.0,
        ffn_residual_scale: float = 1.0,
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
                    num_heads,
                    ffn_dim,
                    dropout,
                    attention_backend,
                    block_size,
                    position_encoding=position_encoding,
                    rope_theta=rope_theta,
                    value_position_encoding=value_position_encoding,
                    attention_logit_scale_multiplier=attention_logit_scale_multiplier,
                    learned_attention_logit_scale=learned_attention_logit_scale,
                    attention_top_k=attention_top_k,
                    attention_residual_scale=attention_residual_scale,
                    ffn_residual_scale=ffn_residual_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        local_valid: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        valid_neighbors: torch.Tensor | None = None,
        block_pair_index: torch.Tensor | None = None,
        local_log_m: torch.Tensor | None = None,
        neighbor_log_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.token(x)
        if self.pos is not None:
            pos = torch.arange(x.shape[1], device=x.device)
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
        return self.head(self.norm(h))
