"""Resample non-16kHz waveform audio in a speech instruction JSONL to 16kHz.

The stage-2 corpus (from the ModelScope dataset) mixes sources whose native
rates differ (AISHELL=16kHz, moss_speech_qa TTS=24kHz, voiceassistant400k=22050Hz).
``train_speech_qwen3.py`` enforces 16kHz input, so this script rewrites the
non-16kHz rows to resampled 16-bit PCM WAV copies under an output audio
directory and repoints their audio field to the new absolute path (source files
are left untouched; 16kHz rows are unchanged).

Accept either a single JSONL (stage-2 ``wav`` rows, or legacy ``audio`` rows)
or a directory of ``{train,dev,...}.jsonl``. The manifest is rewritten in place;
every other field is preserved.

Output layout: ``<out>/<split>/<row_index>.wav`` (16-bit PCM, target rate).
Idempotent: rows already pointing at 16kHz audio are skipped.

Requires: soundfile (read/write) and soxr (fast high-quality resample); falls
back to scipy.signal.resample_poly if soxr is unavailable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import soxr
    HAS_SOXR = True
except ImportError:  # pragma: no cover - optional fast resampler
    HAS_SOXR = False
    from scipy.signal import resample_poly

TARGET = 16000


def _resample_1d(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if HAS_SOXR:
        return soxr.resample(y, sr, target)
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(y, target // g, sr // g)


def resample_audio(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return _resample_1d(y, sr, target)
    return np.stack([_resample_1d(y[:, c], sr, target) for c in range(y.shape[1])], axis=1)


def audio_field(record: dict) -> str:
    """Field name holding the audio path: stage-2 ``wav`` or legacy ``audio``."""
    if str(record.get("wav", "")).strip():
        return "wav"
    return "audio"


def process_manifest(manifest: Path, out_dir: Path, target: int, overwrite: bool,
                     split: str | None = None) -> dict:
    split = split or manifest.stem
    rows: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    resampled = kept = written = 0
    started = time.time()
    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    for idx, record in enumerate(rows):
        field = audio_field(record)
        audio = str(record.get(field, "")).strip()
        if not audio:
            out_rows.append(record)
            continue
        info = sf.info(audio)
        if info.samplerate == target:
            kept += 1
            out_rows.append(record)
            continue
        data, sr = sf.read(audio, dtype="float32", always_2d=False)
        resampled_audio = resample_audio(data, sr, target)
        dest = split_out / f"{idx:06d}.wav"
        if not dest.exists() or overwrite:
            sf.write(str(dest), resampled_audio, target, subtype="PCM_16")
            written += 1
        rec = dict(record)
        rec[field] = str(dest.resolve())
        out_rows.append(rec)
        resampled += 1

    with manifest.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    return {
        "split": split, "rows": n, "kept_16k": kept, "resampled": resampled,
        "written": written, "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True,
                   help="a JSONL manifest, or a directory with {train,dev}.jsonl")
    p.add_argument("--sr", type=int, default=TARGET, help="target sample rate (default 16000)")
    p.add_argument("--out-audio-dir", type=Path, default=None,
                   help="where to write resampled wavs (default <data-dir>/resampled_audio)")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite resampled wavs even if they already exist")
    p.add_argument("--splits", default="train,dev",
                   help="split names to process when --data is a directory")
    args = p.parse_args()

    if args.data.is_file():
        manifests = [(args.data, args.data.stem)]
        base = args.data.parent
    else:
        manifests = [(args.data / f"{s.strip()}.jsonl", s.strip())
                     for s in args.splits.split(",") if s.strip()]
        base = args.data

    out_dir = args.out_audio_dir or (base / "resampled_audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"resampler soxr={HAS_SOXR} target={args.sr} out={out_dir}", flush=True)

    for manifest, split in manifests:
        if not manifest.exists():
            print(f"! missing {manifest}", flush=True)
            continue
        stats = process_manifest(manifest, out_dir, args.sr, args.overwrite, split)
        print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
