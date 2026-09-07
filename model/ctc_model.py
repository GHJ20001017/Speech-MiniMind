"""CTC model built on the Tiny Conformer encoder."""

from __future__ import annotations

import torch
from torch import nn

from .conformer import TinyConformer


class TinyConformerCTC(nn.Module):
    def __init__(self, vocab_size: int, blank_id: int = 0, **encoder_kwargs: object) -> None:
        super().__init__()
        self.encoder = TinyConformer(**encoder_kwargs)
        self.ctc_head = nn.Linear(encoder_kwargs.get("hidden_dim", 256), vocab_size)
        self.blank_id = blank_id

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.ctc_head(self.encoder(features, padding_mask))
