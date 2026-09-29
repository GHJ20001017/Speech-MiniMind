"""Datasets for the ASR/TTS/audio-continuation Route-B pretraining mix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskAudioSample:
    def __init__(self, task: str, codes: torch.Tensor, text: str = "", prompt_codes: torch.Tensor | None = None,
                 messages: list[dict[str, str]] | None = None) -> None:
        self.task = task
        self.codes = codes
        self.prompt_codes = prompt_codes
        self.text = text
        self.messages = messages


def _load_codes(path: Path) -> torch.Tensor:
    array = np.load(path)
    if array.ndim == 1:
        array = array[None, :]
    codes = torch.from_numpy(np.asarray(array, dtype=np.int64))
    if codes.dim() != 2:
        raise ValueError(f"expected code array (Q,T), got {tuple(codes.shape)}: {path}")
    return codes


class EmiliaTaskDataset(Dataset):
    """One task view over an Emilia code manifest.

    The same utterance/text pair is exposed as three different causal tasks;
    task identity is kept in the dataset so a DataLoader batch is homogeneous.
    """

    def __init__(self, manifest: Path, task: str, max_frames: int | None = None,
                 min_frames: int = 8, continuation_prefix_ratio: tuple[float, float] = (0.25, 0.75),
                 lang_filter: str | None = None, limit: int = 0, seed: int = 7) -> None:
        if task not in {"asr", "tts", "audio_lm"}:
            raise ValueError(f"unknown task: {task}")
        self.manifest, self.task = Path(manifest), task
        # Legacy max_frames is accepted for callers but never crops paired audio.
        self.min_frames = min_frames
        self.prefix_ratio, self.seed, self.epoch = continuation_prefix_ratio, seed, 0
        with self.manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if lang_filter:
            self.rows = [row for row in self.rows if row.get("lang") == lang_filter]
        if limit:
            self.rows = self.rows[:limit]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MultiTaskAudioSample:
        row = self.rows[index]
        code_path = Path(row["codes"])
        if not code_path.is_absolute():
            code_path = self.manifest.parent / code_path
        codes = _load_codes(code_path)
        if codes.size(1) < self.min_frames:
            raise ValueError(f"row {index} is too short: {codes.size(1)} frames")
        if self.task == "audio_lm":
            low = max(1, int(codes.size(1) * self.prefix_ratio[0]))
            high = min(codes.size(1) - 1, int(codes.size(1) * self.prefix_ratio[1]))
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index + 17)
            split = int(torch.randint(low, max(low + 1, high + 1), (1,), generator=generator))
            return MultiTaskAudioSample(self.task, codes[:, split:].clone(), row.get("text", ""), codes[:, :split].clone())
        return MultiTaskAudioSample(self.task, codes, row.get("text", ""))


def multitask_collate(samples: list[MultiTaskAudioSample]) -> list[MultiTaskAudioSample]:
    if not samples:
        return samples
    tasks = {sample.task for sample in samples}
    if len(tasks) != 1:
        raise ValueError(f"a batch must contain one task, got {tasks}")
    return samples
