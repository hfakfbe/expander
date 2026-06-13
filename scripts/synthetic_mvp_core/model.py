from __future__ import annotations

import torch
import torch.nn as nn

from .attention import local_blockpair_attention, local_cross_attention


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_backend: str,
        block_size: int,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

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
            )
        else:
            raise ValueError(f"unknown attention backend: {self.attention_backend}")
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
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedSelfAttention(d_model, num_heads, dropout, attention_backend, block_size)
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
        x = x + self.attn(
            self.ln1(x),
            mask,
            local_valid,
            neighbors,
            valid_neighbors,
            block_pair_index,
            local_log_m,
            neighbor_log_m,
        )
        return x + self.ffn(self.ln2(x))

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
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(d_model, num_heads, ffn_dim, dropout, attention_backend, block_size)
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
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.token(x) + self.pos(pos)[None, :, :]
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
