"""Frozen neural audio codec adapters for the Route-B audio-native LLM.

Route B turns a waveform into a **discrete codebook sequence**, lets an
audio-only LLM model that sequence, and turns the generated codes back into a
waveform.  The codec that performs the waveform <-> codes mapping is a frozen,
pre-trained, external model: this project never trains it.

This module mirrors the design of :mod:`model.frozen_encoder` (the Route-A
continuous encoder adapters) so both routes share one mental model::

    codec = build_frozen_audio_codec("mimi", model_id=..., device=device)
    codes, code_lengths = codec.encode(waveforms, lengths, sample_rate)
    # codes:        (B, num_codebooks, T_c)  int64
    # code_lengths: (B,)                     real frame count per utterance
    waveforms, lengths = codec.decode(codes, code_lengths)
    # waveforms:    (B, N) float32 in [-1, 1]

Supported backends (lazy imports so the repo stays importable without them):

* :class:`MimiCodec` - Kyutai ``Mimi`` (8 codebooks, 1.25 kbps, 12.5 Hz frames,
  24 kHz).  This is also what the MiniMind-O ``sft_a2a`` dataset was tokenised
  with, so it is the default Route-B codec.
* :class:`EncodecCodec` - Meta ``EnCodec`` 24 kHz (RVQ-8 by default).  Kept as
  the comparison backend from the training plan.

Both are exposed through the same interface, so the trainer never branches on
the backend.  ``num_codebooks == 1`` is a valid configuration (single-codebook
codecs such as WavTokenizer can be wrapped the same way): the LLM layer treats
"one codebook" and "N codebooks" identically by flattening frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import gcd

import numpy as np
import torch


def _resample_1d(y: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample one channel from ``source_rate`` to ``target_rate``.

    Prefers ``soxr`` (fast, high quality), falls back to ``scipy.signal``, and
    finally to linear interpolation so the module never hard-depends on either.
    """
    if source_rate == target_rate:
        return y
    try:
        import soxr

        return soxr.resample(y, source_rate, target_rate).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly

        factor = gcd(source_rate, target_rate)
        return resample_poly(y, target_rate // factor, source_rate // factor).astype(np.float32)
    except ImportError:
        new_length = max(1, round(len(y) * target_rate / source_rate))
        old = np.linspace(0.0, 1.0, len(y), endpoint=False)
        new = np.linspace(0.0, 1.0, new_length, endpoint=False)
        return np.interp(new, old, y).astype(np.float32)


def resample_batch(
    waveforms: torch.Tensor,
    lengths: torch.Tensor,
    source_rate: int,
    target_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a padded ``(B, N)`` waveform batch on CPU, keeping real lengths.

    Codecs run at a fixed rate (Mimi/EnCodec: 24 kHz) while our corpora are
    16 kHz (AISHELL) or 24 kHz (Qwen3-TTS).  Resampling here keeps every
    downstream consumer rate-agnostic.
    """
    if source_rate == target_rate:
        return waveforms, lengths
    out: list[torch.Tensor] = []
    out_lengths: list[int] = []
    for index in range(waveforms.size(0)):
        real = int(lengths[index])
        samples = waveforms[index, :real].cpu().numpy().astype(np.float32)
        resampled = _resample_1d(samples, source_rate, target_rate)
        out.append(torch.from_numpy(resampled))
        out_lengths.append(resampled.size)
    padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
    return padded, torch.tensor(out_lengths, dtype=torch.long)


class FrozenAudioCodec(ABC):
    """Common interface shared by every frozen audio-codec backend."""

    num_codebooks: int
    """Number of residual codebooks ``Q`` (1 for a single-codebook codec)."""

    codebook_size: int
    """Vocabulary size of one codebook (codes are ``0 .. codebook_size - 1``)."""

    sample_rate: int
    """Waveform sample rate of :meth:`decode` (Hz)."""

    frame_rate_hz: float
    """Code frames produced per second (Mimi: 12.5, EnCodec 24k: 75)."""

    def train(self, mode: bool = True) -> "FrozenAudioCodec":
        """Forward ``train/eval`` to the wrapped module (stays frozen anyway)."""
        self._engine.train(mode)
        return self

    def eval(self) -> "FrozenAudioCodec":
        self._engine.eval()
        return self

    @property
    def samples_per_frame(self) -> int:
        return int(round(self.sample_rate / self.frame_rate_hz))

    @abstractmethod
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms into discrete codes.

        Args:
            waveforms: ``(B, N)`` float32 in [-1, 1].
            lengths: ``(B,)`` real sample counts per utterance.
            sample_rate: rate of ``waveforms`` (defaults to the codec's own
                rate).  Mismatched rates are resampled internally.

        Returns:
            codes: ``(B, num_codebooks, T_c)`` int64.
            code_lengths: ``(B,)`` real frame counts after padding removal.
        """

    @abstractmethod
    def decode(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode discrete codes back into waveforms.

        Args:
            codes: ``(B, num_codebooks, T_c)`` int64.
            code_lengths: ``(B,)`` real frame counts (``None`` = full length).

        Returns:
            waveforms: ``(B, N)`` float32 in [-1, 1].
            lengths: ``(B,)`` real sample counts.
        """

    def frames_to_samples(self, code_lengths: torch.Tensor) -> torch.Tensor:
        return (code_lengths * self.samples_per_frame).clamp_min(1)


class MimiCodec(FrozenAudioCodec):
    """Kyutai ``Mimi`` codec adapter (8 codebooks, 12.5 Hz, 24 kHz).

    ``Mimi`` is the codec used by MiniMind-O, so tokenising our own audio with
    it keeps us bit-compatible with the public ``sft_a2a`` dataset.
    """

    def __init__(
        self,
        model_id: str = "kyutai/mimi",
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            from transformers import AutoFeatureExtractor, MimiModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "MimiCodec requires 'transformers' with Mimi support "
                "(transformers>=4.42). Install with: python -m pip install -r requirements.txt"
            ) from error

        self._device = torch.device(device)
        self._model = MimiModel.from_pretrained(model_id).to(self._device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._engine = self._model

        config = self._model.config
        self.num_codebooks = int(getattr(config, "num_quantizers", 8) or 8)
        self.codebook_size = int(getattr(config, "codebook_size", 2048) or 2048)

        feature_extractor = None
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        except Exception:  # noqa: BLE001 - optional metadata source
            feature_extractor = None
        self.sample_rate = int(
            getattr(feature_extractor, "sampling_rate", None)
            or getattr(config, "sampling_rate", 24000)
            or 24000
        )
        # Mimi packs 1920 samples (80 ms) per frame at 24 kHz => 12.5 frames/s.
        self.frame_rate_hz = self.sample_rate / 1920.0

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rate = int(sample_rate or self.sample_rate)
        waveforms, lengths = resample_batch(waveforms, lengths, rate, self.sample_rate)
        input_values = waveforms.to(self._device).unsqueeze(1)  # (B, 1, N)
        padding_mask = (
            torch.arange(input_values.size(-1), device=self._device)[None, :]
            < lengths.to(self._device)[:, None]
        )
        output = self._model.encode(input_values, padding_mask=padding_mask)
        codes = output.audio_codes  # (B, Q, T_c)
        if codes.dtype != torch.long:
            codes = codes.long()
        frame_lengths = self._frame_lengths(codes.size(2), lengths)
        codes = codes.masked_fill(
            torch.arange(codes.size(2), device=codes.device)[None, None, :]
            >= frame_lengths[:, None, None].to(codes.device),
            0,
        )
        return codes.cpu(), frame_lengths

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = codes.to(self._device)
        if code_lengths is None:
            code_lengths = torch.full(
                (codes.size(0),), codes.size(2), dtype=torch.long
            )
        code_lengths = code_lengths.to(self._device)
        padding_mask = (
            torch.arange(codes.size(-1), device=self._device)[None, :]
            < code_lengths[:, None]
        ).unsqueeze(1)  # (B, 1, T_c)
        output = self._model.decode(codes, padding_mask=padding_mask)
        audio_values = output.audio_values.squeeze(1)  # (B, N)
        lengths = self.frames_to_samples(code_lengths.cpu())
        return audio_values.cpu().clamp(-1.0, 1.0), lengths

    def _frame_lengths(self, frames: int, lengths: torch.Tensor) -> torch.Tensor:
        """Map waveform sample counts to Mimi frame counts (80 ms per frame)."""
        return (lengths.float() / self.sample_rate * self.frame_rate_hz).ceil().long().clamp(1, frames)


class EncodecCodec(FrozenAudioCodec):
    """Meta ``EnCodec`` 24 kHz adapter (RVQ, default 8 codebooks, 75 Hz)."""

    def __init__(
        self,
        model_id: str = "facebook/encodec_24khz",
        device: str | torch.device = "cpu",
        num_codebooks: int | None = None,
    ) -> None:
        try:
            from transformers import AutoFeatureExtractor, EncodecModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "EncodecCodec requires 'transformers' with EnCodec support. "
                "Install with: python -m pip install -r requirements.txt"
            ) from error

        self._device = torch.device(device)
        self._model = EncodecModel.from_pretrained(model_id).to(self._device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._engine = self._model

        config = self._model.config
        self.num_codebooks = int(
            num_codebooks or getattr(config, "num_quantizers", 8) or 8
        )
        self.codebook_size = int(getattr(config, "codebook_size", 1024) or 1024)

        feature_extractor = None
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        except Exception:  # noqa: BLE001 - optional metadata source
            feature_extractor = None
        self.sample_rate = int(
            getattr(feature_extractor, "sampling_rate", None)
            or getattr(config, "sampling_rate", 24000)
            or 24000
        )
        hop = int(getattr(config, "hop_length", 320) or 320)
        self.frame_rate_hz = self.sample_rate / hop
        self._use_scales = True

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rate = int(sample_rate or self.sample_rate)
        waveforms, lengths = resample_batch(waveforms, lengths, rate, self.sample_rate)
        input_values = waveforms.to(self._device).unsqueeze(1)
        padding_mask = (
            torch.arange(input_values.size(-1), device=self._device)[None, :]
            < lengths.to(self._device)[:, None]
        )
        output = self._model.encode(input_values, padding_mask=padding_mask)
        codes = output.audio_codes
        if codes.dim() == 4:  # (B, Q, 1, T) on some versions
            codes = codes.squeeze(2)
        codes = codes[:, : self.num_codebooks, :]
        if codes.dtype != torch.long:
            codes = codes.long()
        frame_lengths = (lengths.float() / self.sample_rate * self.frame_rate_hz).ceil().long()
        frame_lengths = frame_lengths.clamp(1, codes.size(2))
        codes = codes.masked_fill(
            torch.arange(codes.size(2), device=codes.device)[None, None, :]
            >= frame_lengths[:, None, None].to(codes.device),
            0,
        )
        return codes.cpu(), frame_lengths

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = codes.to(self._device)
        batch = codes.size(0)
        if code_lengths is None:
            code_lengths = torch.full((batch,), codes.size(2), dtype=torch.long)
        code_lengths = code_lengths.to(self._device)
        padding_mask = (
            torch.arange(codes.size(-1), device=self._device)[None, :]
            < code_lengths[:, None]
        )
        # EnCodec needs one scale entry per item; `None` means "no rescaling",
        # which is what we want for codes we produced ourselves at full scale.
        output = self._model.decode(
            codes.unsqueeze(2) if codes.dim() == 3 else codes,
            [None] * batch,
            padding_mask=padding_mask,
        )
        audio_values = output.audio_values
        if audio_values.dim() == 3:
            audio_values = audio_values.squeeze(1)
        lengths = self.frames_to_samples(code_lengths.cpu())
        return audio_values.cpu().clamp(-1.0, 1.0), lengths


def build_frozen_audio_codec(
    codec_type: str,
    model_id: str | None = None,
    device: str | torch.device = "cpu",
    num_codebooks: int | None = None,
) -> FrozenAudioCodec:
    """Return the codec adapter selected by ``--codec-type``.

    Supported values: ``mimi`` (default, MiniMind-O compatible) and ``encodec``.
    """
    key = (codec_type or "mimi").lower().replace("-", "_")
    if key in ("mimi", "kyutai_mimi"):
        return MimiCodec(model_id=model_id or "kyutai/mimi", device=device)
    if key in ("encodec", "encodec_24khz"):
        return EncodecCodec(
            model_id=model_id or "facebook/encodec_24khz",
            device=device,
            num_codebooks=num_codebooks,
        )
    raise ValueError(
        f"unknown --codec-type '{codec_type}'. Supported: mimi, encodec"
    )
