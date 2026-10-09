"""Frozen SenseVoice input and Mimi output frontend. Base format-v5 weights remain independently valid."""
from pathlib import Path
import json

import torch
from torch import nn

from trainer.train_audio_multitask import load_multitask_checkpoint, save_checkpoint

LEGACY_S2S_METADATA = {
    "s2s_format_version": 2, "frontend": "sensevoice_layernorm_linear_gelu_linear",
    "prompt_layout": "user_audio_text_mixture_empty_assistant_v2", "codec": "mimi",
    "num_codebooks": 8, "codebook_size": 2048,
    "sample_rate": 16000, "encoder_frozen": True,
    "stopped_stream_feedback": "first_stop_then_pad",
}


S2S_METADATA = dict(
    LEGACY_S2S_METADATA, s2s_format_version=3,
    prompt_layout="full_history_last_user_mixture_empty_assistant_v3",
    data_policy="minimind_o_f900448c_direct_parquet",
    tone_conditioning="none", scheduled_sampling=0.05,
)


def load_encoder(path, device):
    from model.frozen_encoder import SenseVoiceFrozenEncoder
    if not Path(path).is_dir():
        raise ValueError("SenseVoice requires an existing local model directory (no automatic download)")
    return SenseVoiceFrozenEncoder(str(path), device=device)


def validate_codes(codes):
    try:
        codes = torch.as_tensor(codes)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Mimi codes must be nonempty integer [8,T] in 0..2047") from error
    if (codes.ndim != 2 or codes.shape[0] != 8 or codes.shape[1] < 1
            or codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or ((codes < 0) | (codes >= 2048)).any()):
        raise ValueError("Mimi codes must be nonempty integer [8,T] in 0..2047")
    return codes.long()


class SpeechInputModel(nn.Module):
    def __init__(self, base, encoder, encoder_path="../model/SenseVoiceSmall", tuning="audio_proj"):
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.encoder_path = str(encoder_path)
        hidden = base.config.hidden_size
        reference = base.get_input_embeddings().weight
        self.frontend = nn.Sequential(nn.LayerNorm(encoder.output_dim),
                                     nn.Linear(encoder.output_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden)).to(reference)
        self.set_tuning(tuning)

    def set_tuning(self, tuning):
        if tuning not in ("audio_proj", "all"):
            raise ValueError("tuning must be audio_proj or all")
        self.tuning = tuning
        self.base.requires_grad_(tuning == "all")
        self.frontend.requires_grad_(True)
        self.encoder._engine.requires_grad_(False).float().eval()
        self.train(self.training)

    @torch.no_grad()
    def encode_waveforms(self, waveforms, augment=False):
        lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
        if not len(waveforms) or (lengths < 1).any():
            raise ValueError("waveforms must be nonempty")
        padded = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        if padded.ndim != 2 or not torch.isfinite(padded).all():
            raise ValueError("waveforms must be finite mono samples")
        self.encoder.eval()
        if augment:
            from dataset.thinker_talker_s2s import augment_fbank
            hidden, sizes = self.encoder.encode(padded.float(), lengths, 16000, feature_augment=augment_fbank)
        else:
            hidden, sizes = self.encoder.encode(padded.float(), lengths, 16000)
        if hidden.ndim != 3 or hidden.shape[0] != len(waveforms) or hidden.shape[2] != self.encoder.output_dim:
            raise ValueError("invalid SenseVoice feature dimensions")
        if len(sizes) != len(waveforms) or (sizes < 1).any() or (sizes > hidden.shape[1]).any():
            raise ValueError("invalid SenseVoice feature lengths")
        return [h[:int(n)].detach() for h, n in zip(hidden, sizes)]

    @property
    def config(self):
        return self.base.config

    @property
    def thinker(self):
        return self.base.thinker

    @property
    def audio_streams(self):
        return self.base.audio_streams

    def train(self, mode=True):
        super().train(mode)
        if self.tuning == "audio_proj":
            self.base.eval()  # retain autograd through the frozen backbone
        self.encoder.eval()
        return self

    def forward_streams(self, input_ids, audio_inputs, *, speech_features=None,
                        speech_mask=None, **kwargs):
        if speech_features is None and speech_mask is None:
            return self.base.forward_streams(input_ids, audio_inputs, **kwargs)
        if (not isinstance(speech_features, (list, tuple)) or len(speech_features) != input_ids.shape[0]
                or speech_mask is None or speech_mask.shape != input_ids.shape
                or speech_mask.dtype != torch.bool or speech_mask.device != input_ids.device):
            raise ValueError("speech features and boolean speech mask must match the input batch")
        embeddings = self.base.get_input_embeddings().weight.new_zeros(
            *input_ids.shape, self.config.hidden_size)
        for i, codes in enumerate(speech_features):
            codes = codes.to(self.frontend[1].weight)
            if codes.ndim != 2 or codes.shape[1] != self.encoder.output_dim or not torch.isfinite(codes).all():
                raise ValueError("invalid SenseVoice features")
            if int(speech_mask[i].sum()) != codes.shape[0]:
                raise ValueError("speech slot count must equal question frame count")
            if codes.shape[0]:
                embeddings[i, speech_mask[i]] = self.frontend(codes.detach())
        if not speech_mask.any():
            return self.base.forward_streams(input_ids, audio_inputs, **kwargs)
        return self.base.forward_streams(input_ids, audio_inputs,
                                        speech_embeddings=embeddings,
                                        speech_mask=speech_mask, **kwargs)


def save_s2s_checkpoint(model, tokenizer, path, epoch=0):
    save_checkpoint(model.base, tokenizer, path, task="s2s", epoch=epoch)
    path = Path(path)
    torch.save(model.frontend.state_dict(), path / "speech_frontend.pt")
    (path / "s2s_metadata.json").write_text(json.dumps(dict(S2S_METADATA, tuning=model.tuning, encoder_path=model.encoder_path,
                                                    acoustic_dim=model.encoder.output_dim), indent=2) + "\n")


def load_s2s_checkpoint(path, device, encoder_path=None, encoder=None):
    path = Path(path)
    if not (path / "s2s_metadata.json").is_file() or not (path / "speech_frontend.pt").is_file():
        raise ValueError("S2S requires trained speech_frontend.pt and s2s_metadata.json")
    metadata = json.loads((path / "s2s_metadata.json").read_text())
    expected = (LEGACY_S2S_METADATA if isinstance(metadata, dict) and metadata.get("s2s_format_version") == 2
                else S2S_METADATA)
    if (not isinstance(metadata, dict)
            or set(metadata) != set(expected) | {"tuning", "encoder_path", "acoustic_dim"}
            or any(metadata.get(k) != v for k, v in expected.items())
            or metadata.get("tuning") not in ("audio_proj", "all")
            or not isinstance(metadata.get("encoder_path"), str) or not metadata["encoder_path"]
            or type(metadata.get("acoustic_dim")) is not int or metadata["acoustic_dim"] < 1):
        raise ValueError("incompatible S2S frontend metadata")
    base, tokenizer = load_multitask_checkpoint(path, device)
    encoder_path = encoder_path or metadata["encoder_path"]
    encoder = encoder or load_encoder(encoder_path, device)
    if encoder.output_dim != metadata["acoustic_dim"]:
        raise ValueError("incompatible SenseVoice acoustic dimension")
    model = SpeechInputModel(base, encoder, encoder_path, metadata["tuning"])
    model.frontend.load_state_dict(torch.load(path / "speech_frontend.pt", map_location=device,
                                              weights_only=True), strict=True)
    return model.eval(), tokenizer
