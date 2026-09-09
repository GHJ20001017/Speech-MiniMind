"""A chunk-based streaming (causal) Conformer for Speech MiniMind.

This is the streaming counterpart to :mod:`model.conformer.TinyConformer`.
The non-streaming version uses bidirectional self-attention and center-padded
convolution, so it needs the whole utterance before emitting anything. This
module keeps the same overall architecture (subsampling -> stacked Conformer
blocks -> linear head) but makes every component causal so a stream of audio
can be processed with bounded latency:

* **Causal subsampling** - two ``Conv2d(kernel=(5, 2), stride=(2, 2))`` layers
  that only ever read the current and past frames (non-overlapping strided
  window in the time axis).
* **Causal depthwise conv** - left-padded ``Conv1d`` (``padding=kernel-1``)
  instead of center padding, so a frame never sees future frames.
* **Chunked causal self-attention** - input is decoded in fixed-size chunks
  ``C`` frames long (in the block/time space). Each chunk attends to the
  previous ``L`` frames (left context) plus the frames earlier within the same
  chunk, through an explicit causal mask. Anything beyond that is invisible,
  which bounds the latency.

Normalization follows the standard streaming-conformer compromise also used by
WeNet: batch/layer norm is applied over the frames that are actually processed
together (the chunk plus its left context). It never reaches across chunk
boundaries into the future, so training and incremental inference stay
consistent.

The resulting encoder can be consumed two ways:

* ``forward(features, ...)`` receives a whole utterance and returns all frames
  at once, but with streaming (causal) semantics - useful for training.
* ``forward_chunk(chunk, cache)`` feeds one chunk at a time and is the
  real-time inference path. ``init_cache`` builds the initial cache.

Latency (in the block / post-subsampling frame space) is ``C`` frames of
processing plus ``L`` frames of actual look-back; the streaming path has no
look-ahead.
"""

from __future__ import annotations

import torch
from torch import nn


class CausalConv2d(nn.Module):
    """Conv2d that is causal along the time (last) axis only.

    The kernel is non-overlapping in time (``kernel_time=2, stride=2``), so
    output frame ``n`` only uses input frames ``2n`` and ``2n-1`` (never the
    future). Frequency uses a normal causal-friendly symmetric pad.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_freq: int = 5) -> None:
        super().__init__()
        # padding: symmetric on frequency (kernel_freq//2), zero on time.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_freq, 2),
            stride=(2, 2),
            padding=(kernel_freq // 2, 0),
        )
        self.activation = nn.ReLU()

    def time_out(self, time_in: int) -> int:
        """Output time length for a causal stride-2 windowed kernel."""
        return (time_in - 1) // 2 + 1 if time_in >= 2 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        return self.activation(self.conv(x))


class CausalConv1d(nn.Module):
    """Depthwise-capable Conv1d that is causal along time via left padding.

    ``padding=kernel_size-1`` on the left, then the trailing ``T`` frames are
    sliced off, so frame ``n`` sees at most frames ``n-kernel+1 .. n``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(x)[..., : x.size(-1)]


class CausalConvolutionModule(nn.Module):
    """Depthwise-separable conv block, everything causal in time."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = CausalConv1d(dim, dim, kernel_size, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x).transpose(1, 2)  # (B, D, T)
        hidden = nn.functional.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.activation(self.batch_norm(self.depthwise(hidden)))
        hidden = self.pointwise_out(hidden).transpose(1, 2)  # (B, T, D)
        return self.dropout(hidden)


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


class StreamingConformerBlock(nn.Module):
    """Conformer block with causal convolution and chunked causal attention."""

    def __init__(self, dim: int = 256, heads: int = 4, ff_expansion: int = 4, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = FeedForward(dim, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = CausalConvolutionModule(dim, kernel_size, dropout)
        self.ffn2 = FeedForward(dim, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def _mask_for(self, n_chunk: int, n_context: int, device: torch.device) -> torch.Tensor:
        """Causal bool mask over the combined ``n_context + n_chunk`` window.

        Returns a square ``(total, total)`` mask where row ``r`` (a query) may
        attend key ``c`` if ``r`` is in the context region (discarded later) or
        ``c <= r`` (causal). PyTorch's attention requires a square 2D mask when
        combined with a ``key_padding_mask``.
        """
        total = n_context + n_chunk
        rows = torch.arange(total, device=device).unsqueeze(1)  # (total, 1)
        cols = torch.arange(total, device=device).unsqueeze(0)  # (1, total)
        causal = cols <= rows
        # Context rows attend everything (their outputs are thrown away anyway);
        # only the chunk rows carry the causal constraint.
        fully_open = rows < n_context
        allow = causal | fully_open
        # ``nn.MultiheadAttention`` treats a 2D bool ``attn_mask`` element as
        # *True == blocked* (it maps them to ``-inf`` additive biases), the
        # opposite of ``allow`` above. Flip so the causal pattern survives the
        # hand-off; otherwise every allowed key is masked and, on CUDA, the
        # softmax can hit an all-``-inf`` row and emit NaN (the CPU math kernel
        # happens not to). See ``merge_masks`` in ``torch.nn.modules.activation``.
        return ~allow  # True == blocked

    def forward(
        self,
        x: torch.Tensor,  # (B, n_context + n_chunk, D) causal combined block input
        n_chunk: int,
        n_context: int,
        padding_mask: torch.Tensor | None = None,  # (B, n_context + n_chunk) True == pad
    ) -> torch.Tensor:
        mask = self._mask_for(n_chunk, n_context, x.device)  # (n_chunk, total) True == blocked
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attn_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attended)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class StreamingConformer(nn.Module):
    """Causal chunk-streaming Conformer encoder.

    Drop-in streaming replacement for :class:`model.conformer.TinyConformer`.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 32,      # frames per chunk in the block (post-subsample) space
        left_context: int = 16,    # block-space frames of look-back cached between chunks
    ) -> None:
        super().__init__()
        if chunk_size < 1 or left_context < 0:
            raise ValueError("chunk_size must be >= 1 and left_context >= 0")

        self.chunk_size = chunk_size
        self.left_context = left_context
        self.hidden_dim = hidden_dim

        self.subsampling = nn.Sequential(
            CausalConv2d(1, hidden_dim),
            CausalConv2d(hidden_dim, hidden_dim),
        )
        freq_after = ((n_mels + 2 * 2 - 5) // 2 + 1)      # 80 -> 40
        freq_after = ((freq_after + 2 * 2 - 5) // 2 + 1)  #      -> 20
        self.frequency_after_subsampling = freq_after
        self.input_projection = nn.Linear(hidden_dim * freq_after, hidden_dim)
        self.layers = nn.ModuleList(
            [StreamingConformerBlock(hidden_dim, heads, 4, 31, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
        """Map feature-frame lengths through the two causal stride-2 time convs.

        Each causal ``kernel=2, stride=2`` time conv maps ``n -> ceil(n/2)``
        (for ``n >= 1``), so two of them map ``n -> ceil(n/4)``.
        """
        return torch.div(lengths + 3, 4, rounding_mode="floor").clamp_min(1)

    def _subsample(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, n_mels) -> (B, T', D), causal (no future)."""
        x = features.transpose(1, 2).unsqueeze(1)  # (B, 1, n_mels, T)
        hidden = self.subsampling(x)               # (B, D, F', T')
        batch, _channels, frequency, time = hidden.shape
        hidden = hidden.permute(0, 3, 1, 2).reshape(batch, time, -1)
        return self.input_projection(hidden)

    def forward(
        self,
        features: torch.Tensor,  # (B, T, n_mels) log-mel frames
        padding_mask: torch.Tensor | None = None,  # (B, T) True == padded frame
    ) -> torch.Tensor:
        """Full-utterance forward with streaming (chunked-causal) semantics.

        Processes the utterance with exactly the same raw-frame chunking and
        left-context cache as :meth:`forward_chunk`, so training results are
        reproducible in incremental inference.
        """
        batch, n_raw, _n_mels = features.shape
        raw_per_chunk = self.chunk_size * 4  # 4x time subsampling
        pad = (-n_raw) % raw_per_chunk
        if pad:
            features = nn.functional.pad(features, (0, 0, 0, pad))
        out_chunks: list[torch.Tensor] = []
        cache = self.init_cache(batch, features.device, features.dtype)
        for start in range(0, features.size(1), raw_per_chunk):
            piece = features[:, start : start + raw_per_chunk]
            out, cache = self.forward_chunk(piece, cache)
            out_chunks.append(out)
        padded_out = torch.cat(out_chunks, dim=1)
        n_block = self.subsampled_lengths(torch.tensor([n_raw], device=features.device)).item()
        return padded_out[:, :n_block]

    def init_cache(self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        """Fresh left-context cache for ``forward_chunk``."""
        if self.left_context <= 0:
            return None
        # Cache stores the previous chunk's projected input (post-subsample,
        # pre-block) hidden frames; zero-filled at the start of the stream.
        return torch.zeros(batch, self.left_context, self.hidden_dim, device=device, dtype=dtype)

    def forward_chunk(
        self,
        chunk: torch.Tensor,  # (B, T_chunk, n_mels) one audio chunk to decode
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode one chunk and return its frames plus the updated cache.

        ``chunk`` must be in the *raw* log-mel space (the caller slices the
        audio stream into chunks of raw frames; the encoder subsamples them
        internally). Returns ``(out, new_cache)``.
        """
        hidden = self._subsample(chunk)  # (B, chunk_len', D)
        combined = hidden if cache is None else torch.cat([cache, hidden], dim=1)
        n_chunk = hidden.size(1)
        n_context = combined.size(1) - n_chunk
        out = combined
        for layer in self.layers:
            out = layer(out, n_chunk, n_context)
        out = self.norm(out)
        if self.left_context > 0 and cache is not None:
            new_cache = combined[:, -self.left_context :].detach()
        else:
            new_cache = cache
        return out[:, n_context:], new_cache