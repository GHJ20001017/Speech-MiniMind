"""Create a tiny, reproducible speech-vs-non-speech dataset from the example WAV."""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np

from analyze_audio import read_wav


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.clip(audio, -1, 1)
    pcm = (values * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def make_non_speech(length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length) / sample_rate
    frequency = rng.uniform(180, 900)
    tone = 0.12 * np.sin(2 * math.pi * frequency * time)
    noise = rng.normal(0, 0.015, length)
    return (tone + noise).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("examples/disgusted_to_happy.wav"))
    parser.add_argument("--output", type=Path, default=Path("data/audio_classification"))
    parser.add_argument("--clips-per-class", type=int, default=40)
    parser.add_argument("--clip-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.clips_per_class < 2 or args.clip_seconds <= 0:
        parser.error("clips-per-class must be >= 2 and clip-seconds must be > 0")

    source, sample_rate = read_wav(args.source)
    clip_length = round(sample_rate * args.clip_seconds)
    if source.size < clip_length:
        parser.error("source audio is shorter than clip-seconds")
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for label, name in [(1, "speech"), (0, "non_speech")]:
        for index in range(args.clips_per_class):
            path = args.output / f"{name}_{index:04d}.wav"
            if label == 1:
                start = int(rng.integers(0, source.size - clip_length + 1))
                clip = source[start : start + clip_length]
            else:
                clip = make_non_speech(clip_length, sample_rate, rng)
            write_wav(path, clip, sample_rate)
            rows.append({"path": str(path), "label": label, "name": name})

    rng.shuffle(rows)
    split = max(1, round(len(rows) * 0.8))
    for filename, subset in [("train.csv", rows[:split]), ("valid.csv", rows[split:])]:
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "name"])
            writer.writeheader()
            writer.writerows(subset)
    print(f"dataset_dir: {args.output}")
    print(f"train_examples: {split}")
    print(f"valid_examples: {len(rows) - split}")
    print(f"clip_shape: ({clip_length} samples, {args.clip_seconds:g} s)")


if __name__ == "__main__":
    main()
