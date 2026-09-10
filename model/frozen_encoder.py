"""Unified frozen acoustic-encoder adapters for Speech-MiniMind.

The projector pipeline needs to consume "speech bytes" from an external
(already pre-trained) acoustic encoder whose weights stay frozen.  Different
backends expose very different contracts:

* :class:`TinyConformerEncoder` - the in-repo Conformer. It ingests log-mel
  frames ``(B, T, n_mels)`` and returns ``(B, T//4, 256)`` after two stride-2
  convolutions. It is cheap and fully offline, so it is the default backend
  and the only one that can run without funasr installed.
* :class:`ParaformerFrozenEncoder` - the FunASR ``paraformer-zh-streaming``
  encoder (:class:`SANMEncoderChunkOpt`). Unlike the Conformer this is the
  front half of a *complete* streaming ASR model; it ingests raw waveforms
  and returns frame-level encoder states ``(B, T', 512)`` at ~60 ms/frame
  (fbank 10 ms hop with LFR m=7/n=6 left-context splicing). It has **no**
  internal temporal subsampling, so the hidden-frame rate is fixed by the
  frontend, not by the encoder weights.

This module normalises both backends behind one interface so the projector
trainer does not care which encoder was used::

    encoder = build_frozen_encoder(encoder_type, ...)
    hidden, hidden_lengths = encoder.encode(waveforms, lengths, sample_rate)
    # hidden:            (B, T', acoustic_dim)
    # hidden_lengths:    (B,)  - real frame counts after the frontend

Note
----
Paraformer is a non-autoregressive streaming ASR trained with a CIF
predictor; the encoder states we expose are the *pre-CIF* acoustic
representations (a temporal continuous representation), which is exactly the
signal a speech-LLM projector wants. It is **not** the Whisper-style
``encoder.encoder`` (no layer-norm into per-token embeddings) - no mapping to
`whisper-base/whisper-small` dims is implied, and ``acoustic_dim`` must be set
to the chosen backend's ``output_dim`` (Paraformer = 512).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FrozenSpeechEncoder(ABC):
    """Common interface shared by every frozen acoustic encoder backend.

    Implementations own their frontend internally so the projector trainer can
    feed raw waveforms and never branch on the backend.
    """

    output_dim: int
    """Per-frame feature dimension of :meth:`encode` output (``acoustic_dim``)."""

    output_frame_shift_ms: float
    """Time span covered by one output frame. Used to align frame rates."""

    def train(self, mode: bool = True) -> "FrozenSpeechEncoder":
        """Delegate to the wrapped torch module (kept frozen either way)."""
        self._engine.train(mode)
        return self

    def eval(self) -> "FrozenSpeechEncoder":
        self._engine.eval()
        return self

    @abstractmethod
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of raw waveforms into frozen frame-level states.

        Args:
            waveforms: ``(B, T_samples)`` float32 in [-1, 1].
            lengths: ``(B,)`` real sample counts per utterance.
            sample_rate: input waveform sample rate (default 16000).

        Returns:
            hidden: ``(B, T', output_dim)`` padded representation.
            hidden_lengths: ``(B,)`` real frame counts after the frontend.
        """


class TinyConformerEncoder(FrozenSpeechEncoder):
    """Wraps the repo's Conformer to yield the same frames as before.

    This keeps the historical behaviour: log-mel ``(B, T, 80)`` is downsampled
    by two valid stride-2 convolutions so ``output_dim == 256`` at ~40 ms/frame.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        hidden_dim: int = 256,
        n_mels: int = 80,
        device: str | torch.device = "cpu",
    ) -> None:
        from model.conformer import TinyConformer
        from model.ctc_model import TinyConformerCTC

        self.output_dim = hidden_dim
        self.output_frame_shift_ms = 40.0
        self._device = torch.device(device)

        if checkpoint is None:
            # standalone encoder without a CTC head / vocab
            self._model = TinyConformer(n_mels=n_mels, hidden_dim=hidden_dim)
        else:
            ckpt = torch.load(checkpoint, map_location=self._device, weights_only=False)
            model = TinyConformerCTC(len(ckpt["vocab"]), hidden_dim=hidden_dim)
            model.load_state_dict(ckpt["model"])
            self._model = model.encoder
        self._model.to(self._device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._engine = self._model

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_log_mel(self._waveform_to_log_mel(waveforms, lengths, sample_rate))

    @torch.no_grad()
    def encode_log_mel(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode log-mel frames ``(B, T, 80)`` directly (back-compat path)."""
        features = features.to(self._device)
        if lengths is None:
            lengths = torch.full(
                (features.size(0),), features.size(1), dtype=torch.long, device=features.device
            )
        lengths = lengths.to(self._device)
        hidden = self._model(features, padding_mask=None)
        hidden_lengths = self._model.subsampled_lengths(lengths).clamp_max(hidden.size(1))
        return hidden, hidden_lengths

    @staticmethod
    def _waveform_to_log_mel(
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        from scripts.analyze_audio import log_mel

        batch = []
        for i in range(waveforms.size(0)):
            samples = waveforms[i, : int(lengths[i])].cpu().numpy()
            # keep this encoder backend independent of torchaudio: reuse repo's numpy path
            feats, _, _, _ = log_mel(samples, sample_rate, 25, 10, 80)
            batch.append(torch.from_numpy(feats.astype("float32")))
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


class ParaformerFrozenEncoder(FrozenSpeechEncoder):
    """FunASR ``paraformer-zh-streaming`` frozen encoder adapter.

    Loads the streaming Paraformer via ``funasr.AutoModel`` and exposes only
    its :class:`SANMEncoderChunkOpt` frame-level encoder output (512 dims).
    The waveform->fbank->LFR frontend (:class:`WavFrontendOnline` with
    ``n_mels=80, frame_shift=10ms, lfr_m=7, lfr_n=6``) is applied internally,
    so hidden states come out at ~60 ms/frame with no encoder-side
    subsampling.

    Requires ``funasr`` (and a downloaded model) at runtime; this module only
    imports funasr lazily so the rest of the project stays importable without
    it.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        model_id: str | None = None,
        device: str | torch.device = "cpu",
        disable_dither: bool = True,
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "ParaformerFrozenEncoder requires 'funasr' (and 'modelscope' for the "
                "default ModelScope source). Install them and re-download the model via "
                "scripts/download_paraformer_streaming.py first."
            ) from error

        self.output_dim = 512
        # fbank: 10 ms hop; LFR m=7/n=6 -> one spliced frame per 6 fbank frames.
        self.output_frame_shift_ms = 60.0
        self._device = torch.device(device)
        self._disable_dither = disable_dither

        infer_kwargs = {"device": str(self._device)}
        if model_id:
            infer_kwargs["model"] = model_id
        elif checkpoint:
            infer_kwargs["model"] = str(checkpoint)
        else:
            # default mirror-first to ModelScope per project convention
            infer_kwargs["model"] = (
                "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
            )

        self._auto = AutoModel(**infer_kwargs)
        core_model = self._auto.model  # ParaformerStreaming (torch module)
        self._encoder = core_model.encoder  # SANMEncoderChunkOpt
        self._encoder.to(self._device).eval()
        for parameter in self._encoder.parameters():
            parameter.requires_grad_(False)
        self._engine = self._encoder

        from funasr.frontends.wav_frontend import WavFrontendOnline

        front_conf = {
            "fs": 16000,
            "window": "hamming",
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": 7,
            "lfr_n": 6,
            "dither": 0.0 if self._disable_dither else 1.0,
        }
        self._frontend = WavFrontendOnline(**front_conf)

    @torch.no_grad()
    def encode(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_rate != 16000:
            raise ValueError(
                f"Paraformer frontend expects 16 kHz waveforms, got {sample_rate} Hz. "
                "Resample before calling encode."
            )
        waveforms = waveforms.to(self._device)
        feats, feats_lengths = self._frontend(waveforms, lengths.to(self._device))
        encoder_out, encoder_out_lens, _ = self._encoder(feats, feats_lengths)
        return encoder_out, encoder_out_lens


def build_frozen_encoder(
    encoder_type: str,
    checkpoint: str | None = None,
    model_id: str | None = None,
    device: str | torch.device = "cpu",
    hidden_dim: int = 256,
) -> FrozenSpeechEncoder:
    """Return the encoder adapter selected by ``--encoder-type``.

    Supported values: ``conformer`` (default, offline, no extra deps) and
    ``paraformer`` (FunASR, ModelScope-first).
    """
    key = (encoder_type or "conformer").lower().replace("-", "_")
    if key in ("conformer", "tiny_conformer", "tinyconformer"):
        return TinyConformerEncoder(checkpoint=checkpoint, hidden_dim=hidden_dim, device=device)
    if key in ("paraformer", "paraformer_streaming", "funasr"):
        return ParaformerFrozenEncoder(checkpoint=checkpoint, model_id=model_id, device=device)
    raise ValueError(
        f"unknown --encoder-type '{encoder_type}'. Supported: conformer, paraformer"
    )