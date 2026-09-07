"""A small frame encoder and classifier for the Chapter 02 tutorial."""

from __future__ import annotations

import torch
from torch import nn


class AcousticEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (batch, time, mel)
        hidden = self.projection(features)
        hidden = self.context(hidden.transpose(1, 2)).transpose(1, 2)
        return self.activation(hidden)


class AudioClassifier(nn.Module):
    def __init__(self, n_mels: int = 80, hidden_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = AcousticEncoder(n_mels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)
