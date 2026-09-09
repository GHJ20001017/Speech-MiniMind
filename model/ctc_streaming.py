"""CTC model built on the streaming (chunk-based) Conformer encoder.

Symmetry counterpart of :class:`model.ctc_model.TinyConformerCTC`, but built
on :class:`model.conformer_streaming.StreamingConformer` so the encoder is
causal and can be decoded incrementally. The checkpoint layout (``model`` /
``vocab``) and the ``encoder`` / ``ctc_head`` split match the non-streaming
version, so a streaming encoder can be swapped into the same training and
evaluation scripts by changing a single import.
"""
from __future__ import annotations

import torch
from torch import nn

from .conformer_streaming import StreamingConformer


class TinyStreamingConformerCTC(nn.Module):
    def __init__(self, vocab_size: int, blank_id: int = 0, **encoder_kwargs: object) -> None:
        super().__init__()
        self.encoder = StreamingConformer(**encoder_kwargs)
        self.ctc_head = nn.Linear(encoder_kwargs.get("hidden_dim", 256), vocab_size)
        self.blank_id = blank_id

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.ctc_head(self.encoder(features, padding_mask))