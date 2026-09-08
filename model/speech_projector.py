"""Small adapters that map acoustic encoder states to LLM embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SpeechProjector(nn.Module):
    """Downsample Conformer states and project them into MiniMind's hidden size.

    Input shape is ``[batch, acoustic_steps, acoustic_dim]``. The convolution
    reduces the number of speech tokens before they are prepended to text
    embeddings, which keeps the causal LM sequence length manageable.
    """

    def __init__(self, acoustic_dim: int = 256, llm_dim: int = 768, stride: int = 4) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be positive")
        self.stride = stride
        self.downsample = nn.Conv1d(acoustic_dim, acoustic_dim, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(acoustic_dim)
        self.projection = nn.Sequential(
            nn.Linear(acoustic_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return ((lengths + self.stride - 1) // self.stride).clamp_min(1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.downsample(hidden.transpose(1, 2)).transpose(1, 2)
        return self.projection(self.norm(hidden))
