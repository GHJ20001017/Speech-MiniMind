"""Datasets and on-the-fly augmentation for Speech-MiniMind training.

Augmentation is applied inside ``__getitem__``. Source audio files are never
modified, so every epoch can see a different acoustic variant.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.analyze_audio import read_wav

SAMPLE_RATE = 16000


def augment_mel_features(features: torch.Tensor, frequency_width: int = 64,
                         time_width: int = 10, probability: float = 0.5) -> torch.Tensor:
    """Apply SpecAugment-style masks to a batched ``(B, T, D)`` feature tensor."""
    if features.dim() != 3:
        raise ValueError(f"expected batched features (B, T, D), got {tuple(features.shape)}")
    masked = features.clone()
    _, time_size, frequency_size = masked.shape
    for batch_index in range(masked.size(0)):
        if random.random() < probability and frequency_size > 1:
            width = random.randint(1, min(frequency_width, frequency_size))
            start = random.randint(0, frequency_size - width)
            masked[batch_index, :, start:start + width] = 0
        if random.random() < probability and time_size > 1:
            width = random.randint(1, min(time_width, time_size))
            start = random.randint(0, time_size - width)
            masked[batch_index, start:start + width, :] = 0
    return masked


@dataclass
class SpeechAugmentConfig:
    """Probabilities and ranges for training-time waveform augmentation."""

    enabled: bool = False
    noise_prob: float = 0.30
    noise_std_min: float = 0.001
    noise_std_max: float = 0.01
    speed_prob: float = 0.50
    speed_min: float = 0.70
    speed_max: float = 1.60
    gain_prob: float = 0.30
    gain_min: float = 0.80
    gain_max: float = 1.20
    time_mask_prob: float = 0.20
    reverb_prob: float = 0.20
    lowpass_prob: float = 0.20


class SpeechWaveformAugmenter:
    """Apply random waveform transforms without changing the source files."""

    def __init__(self, config: SpeechAugmentConfig):
        self.config = config

    def __call__(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if not self.config.enabled:
            return waveform.astype(np.float32, copy=False)
        wav = waveform.astype(np.float32, copy=True)
        c = self.config
        if random.random() < c.speed_prob:
            speed = random.uniform(c.speed_min, c.speed_max)
            old = np.linspace(0.0, 1.0, len(wav), endpoint=False)
            new = np.linspace(0.0, 1.0, max(1, round(len(wav) / speed)), endpoint=False)
            wav = np.interp(new, old, wav).astype(np.float32)
        if random.random() < c.noise_prob:
            wav += np.random.randn(len(wav)).astype(np.float32) * random.uniform(c.noise_std_min, c.noise_std_max)
        if random.random() < c.gain_prob:
            wav *= random.uniform(c.gain_min, c.gain_max)
        if random.random() < c.time_mask_prob and len(wav) > sample_rate:
            width = max(1, sample_rate // 4)
            start = random.randrange(0, len(wav) - width + 1)
            wav[start:start + width] = 0.0
        if random.random() < c.lowpass_prob:
            kernel = random.choice((3, 5, 7))
            wav = np.convolve(wav, np.ones(kernel, dtype=np.float32) / kernel, mode="same")
        if random.random() < c.reverb_prob:
            ir_len = max(2, int(sample_rate * random.uniform(0.05, 0.20)))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0.0, 10.0, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode="same").astype(np.float32)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)


class SpeechInstructionDataset(Dataset):
    """Load ``audio,instruction,answer`` JSONL with optional online augmentation."""

    def __init__(self, manifest: Path, augment: SpeechAugmentConfig | None = None,
                 lang_filter: str | None = None, limit: int = 0) -> None:
        self.manifest = Path(manifest)
        self.rows: list[dict[str, str]] = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if lang_filter and record.get("lang") != lang_filter:
                    continue
                audio = str(record.get("audio", "")).strip()
                instruction = str(record.get("instruction", "")).strip()
                answer = str(record.get("answer", "")).strip()
                if audio and answer:
                    self.rows.append({"audio": audio, "instruction": instruction, "answer": answer})
        if limit:
            self.rows = self.rows[:limit]
        self.augmenter = SpeechWaveformAugmenter(augment or SpeechAugmentConfig())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["audio"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        waveform, rate = read_wav(path)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, got {rate} (path={path})")
        waveform = self.augmenter(waveform, rate)
        return torch.from_numpy(waveform), rate, row["instruction"], row["answer"]


class AishellCSVDataset(Dataset):
    """Load legacy AISHELL CSV rows with the same online waveform augmentation."""

    def __init__(self, manifest: Path, prompt: str, augment: SpeechAugmentConfig | None = None) -> None:
        with Path(manifest).open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.prompt = prompt
        self.augmenter = SpeechWaveformAugmenter(augment or SpeechAugmentConfig())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        waveform, rate = read_wav(Path(row["path"]))
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, got {rate}")
        waveform = self.augmenter(waveform, rate)
        return torch.from_numpy(waveform), rate, self.prompt, "".join(row["text"].split()), row["path"]
