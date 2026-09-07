"""A compact, readable Conformer implementation for Speech MiniMind."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x).transpose(1, 2)
        hidden = nn.functional.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.depthwise(hidden)
        hidden = self.activation(self.batch_norm(hidden))
        hidden = self.pointwise_out(hidden).transpose(1, 2)
        return self.dropout(hidden)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, ff_expansion: int = 4, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = FeedForward(dim, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ffn2 = FeedForward(dim, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attn_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)
        x = x + self.attn_dropout(attended)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class TinyConformer(nn.Module):
    """4-layer, 256-hidden Conformer encoder (~7M parameters without CTC head)."""

    def __init__(self, n_mels: int = 80, hidden_dim: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        frequency_after_subsampling = (n_mels - 3) // 2 + 1
        frequency_after_subsampling = (frequency_after_subsampling - 3) // 2 + 1
        self.input_projection = nn.Linear(hidden_dim * frequency_after_subsampling, hidden_dim)
        self.layers = nn.ModuleList([ConformerBlock(hidden_dim, heads, 4, 31, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.subsampling(features.unsqueeze(1))
        batch, channels, time, frequency = hidden.shape
        hidden = hidden.transpose(1, 2).contiguous().view(batch, time, channels * frequency)
        hidden = self.input_projection(hidden)
        if padding_mask is not None:
            padding_mask = padding_mask[:, : hidden.size(1) * 4 : 4]
            padding_mask = padding_mask[:, : hidden.size(1)]
        for layer in self.layers:
            hidden = layer(hidden, padding_mask)
        return self.norm(hidden)
