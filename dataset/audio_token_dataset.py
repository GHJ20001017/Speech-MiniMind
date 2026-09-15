"""Datasets for the Route-B audio-native LLM (discrete codebook).

Two tasks share the same on-disk shape:

* **B0 - pure audio LM**: every row is a single utterance; the whole code
  sequence is the target (audio-only next-token prediction).
* **B1/B2 - speech-to-speech**: every row pairs a *prompt* utterance with an
  *answer* utterance; only the answer codes are supervised.

Both read a JSONL manifest whose rows point at ``.npy`` code shards produced by
``scripts/prepare_speech_to_speech.py`` (for the public MiniMind-O
``sft_a2a`` parquet) or ``scripts/cache_audio_tokens.py`` (for our own audio run
through a frozen codec).  Keeping codes on disk means the codec is never run
during training - re-encoding every epoch would dominate the step time.

Row schema (``s2s``)::

    {"prompt_codes": "codes/train/000123_p.npy",   # (Q, M) int16, may be null
     "answer_codes": "codes/train/000123_a.npy",   # (Q, K) int16
     "task": "speech_qa", "source": "minimind_o_sft_a2a", "lang": "zh"}

Row schema (``audio_lm``)::

    {"codes": "codes/train/000123.npy", "source": "aishell1", "lang": "zh"}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model.audio_lm import AudioSample


def load_codes(path: str | Path | None) -> torch.Tensor | None:
    """Load a ``(Q, T)`` int64 code tensor from an ``.npy`` shard."""
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"code shard not found: {resolved}")
    array = np.load(resolved)
    if array.ndim == 1:  # tolerate (T,) single-codebook shards
        array = array[None, :]
    return torch.from_numpy(np.asarray(array, dtype=np.int64))


def read_manifest(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(manifest: Path, value: str | None) -> str | None:
    """Resolve a manifest-relative path against the manifest's directory."""
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return str(path)


def code_dropout(
    codes: torch.Tensor,
    probability: float,
    codebook_size: int,
    span: int = 3,
) -> torch.Tensor:
    """Randomly corrupt prompt codes to make the model robust to encoder noise.

    Applied to the *prompt* only - the supervised target is never touched.  A
    dropped span is replaced by uniformly sampled valid codes (``0 ..
    codebook_size - 1``) for the same codebook, so vocabulary range is preserved.
    """
    if probability <= 0:
        return codes
    out = codes.clone()
    frames = out.size(1)
    for start in range(frames):
        if torch.rand(1).item() < probability:
            end = min(frames, start + span)
            out[:, start:end] = torch.randint(0, codebook_size, out[:, start:end].shape)
    return out


class SpeechToSpeechDataset(Dataset):
    """``(prompt, answer)`` code pairs; only the answer is supervised."""

    def __init__(
        self,
        manifest: Path,
        max_answer_frames: int = 750,
        max_prompt_frames: int = 750,
        min_answer_frames: int = 1,
        code_dropout_prob: float = 0.0,
        codebook_size: int = 2048,
        lang_filter: str | None = None,
        limit: int = 0,
    ) -> None:
        self.manifest = Path(manifest)
        self.max_answer_frames = max_answer_frames
        self.max_prompt_frames = max_prompt_frames
        self.min_answer_frames = min_answer_frames
        self.code_dropout_prob = code_dropout_prob
        self.codebook_size = codebook_size
        self.rows = read_manifest(self.manifest)
        if lang_filter:
            self.rows = [r for r in self.rows if r.get("lang") == lang_filter]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioSample:
        row = self.rows[index]
        answer = load_codes(resolve(self.manifest, row.get("answer_codes")))
        if answer is None:
            raise ValueError(f"row {index} has no answer codes")
        if answer.size(1) > self.max_answer_frames:
            answer = answer[:, : self.max_answer_frames]
        if answer.size(1) < self.min_answer_frames:
            raise ValueError(f"row {index} answer is too short ({answer.size(1)} frames)")

        prompt = load_codes(resolve(self.manifest, row.get("prompt_codes")))
        if prompt is not None:
            if prompt.size(1) > self.max_prompt_frames:
                prompt = prompt[:, : self.max_prompt_frames]
            if self.code_dropout_prob > 0:
                prompt = code_dropout(prompt, self.code_dropout_prob, self.codebook_size)
        return AudioSample(prompt_codes=prompt, answer_codes=answer)


class AudioLMDataset(Dataset):
    """Pure audio-language-model rows: the whole utterance is the target."""

    def __init__(
        self,
        manifest: Path,
        max_frames: int = 1000,
        min_frames: int = 4,
        lang_filter: str | None = None,
        limit: int = 0,
    ) -> None:
        self.manifest = Path(manifest)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.rows = read_manifest(self.manifest)
        if lang_filter:
            self.rows = [r for r in self.rows if r.get("lang") == lang_filter]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioSample:
        row = self.rows[index]
        codes = load_codes(resolve(self.manifest, row.get("codes")))
        if codes is None:
            raise ValueError(f"row {index} has no codes")
        if codes.size(1) > self.max_frames:
            # Random crop for long utterances so one sample stays one sequence.
            start = int(torch.randint(0, codes.size(1) - self.max_frames + 1, (1,)))
            codes = codes[:, start : start + self.max_frames]
        if codes.size(1) < self.min_frames:
            raise ValueError(f"row {index} is too short ({codes.size(1)} frames)")
        return AudioSample(prompt_codes=None, answer_codes=codes)


def s2s_collate(samples: list[AudioSample]) -> list[AudioSample]:
    """Training batches are built in the trainer; keep raw samples here."""
    return samples
