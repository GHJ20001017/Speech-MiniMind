"""Generate real Chinese instruction audio for moss_speech_qa via Qwen3-TTS.

The moss_speech_qa dataset (derived from fnlp/moss-003-sft-data) stores each
sample's `audio` field as a placeholder path like `audio/000000.wav`, but the
actual WAV files were never created. This script runs Qwen3-TTS (CustomVoice,
12Hz) on each sample's `instruction` text, writes a 16-bit PCM WAV to
`audio/<index>.wav` (resolving the relative path against the JSONL manifest's
parent directory), and leaves every other JSONL field untouched.

Design notes:
- Only the `instruction` text is synthesized (single-turn), matching the
  text-supervised projector training path in train_speech_projector.py.
- `read_wav` in analyze_audio.py requires 16-bit PCM WAV, so results are
  written with soundfile subtype="PCM_16".
- Output audio path stays relative (`audio/NNNNNN.wav`) so both this machine
  and the remote host resolve it identically against the same manifest parent.
- Rows missing an `instruction` or an existing wav file are skipped/recorded.

Model/environment:
- Model dir: /gpu3/guhj/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Python env: /gpu3/guhj/envs/speech_to_speech_system/bin/python
- flash-attn is not installed on the host, so attn_implementation="sdpa" is used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

SUPPORTED_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to a moss_speech_qa split (train.jsonl or dev.jsonl).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/gpu3/guhj/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Path to the local Qwen3-TTS CustomVoice model directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to load the model on (one of the idle GPUs).",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Serena",
        choices=SUPPORTED_SPEAKERS,
        help="Chinese voice/speaker preset (default: Serena, warm female).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="Target TTS language for generate_custom_voice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, process only the first N rows (for a quick smoke test).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional per-row sleep (s) to reduce contention/heat on the GPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate audio even if the target wav file already exists.",
    )
    return parser.parse_args()


def load_rows(jsonl_path: Path) -> list[tuple[int, dict]]:
    """Return [(index, record), ...] in file order with stable zero-padded indices."""
    rows: list[tuple[int, dict]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((line_index, record))
    return rows


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Qwen3-TTS requires a GPU.")
    if args.speaker not in SUPPORTED_SPEAKERS:
        raise ValueError(f"Unknown speaker: {args.speaker}")

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn not installed on the host
    )

    rows = load_rows(args.jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest_dir = args.jsonl.parent
    audio_dir = manifest_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = 0, 0, 0
    failures: list[tuple[int, str]] = []
    started = time.time()

    for index, record in rows:
        audio_rel = str(record.get("audio", "")).strip()
        instruction = str(record.get("instruction", "")).strip()

        audio_path = Path(audio_rel)
        if not audio_path.is_absolute():
            audio_path = manifest_dir / audio_path
            audio_rel = str(audio_path.relative_to(manifest_dir))

        if not instruction:
            skipped += 1
            continue

        if audio_path.exists() and not args.overwrite:
            skipped += 1
            continue

        audio_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=instruction,
                language=args.language,
                speaker=args.speaker,
            )
            audio = wavs[0]
            # Normalize naïvely only if needed; keep it minimal to match the
            # model output. Qwen3-TTS typically returns a clean waveform already.
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Write 16-bit PCM WAV (required by analyze_audio.read_wav).
            sf.write(str(audio_path), audio, sample_rate, subtype="PCM_16")
            done += 1
        except Exception as exc:  # noqa: BLE001 - keep going on per-row failures
            failures.append((index, str(exc)))
            failed += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

        if done % 20 == 0:
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-9)
            print(
                f"[progress] done={done} skipped={skipped} failed={failed} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"[summary] rows={len(rows)} done={done} skipped={skipped} "
        f"failed={failed} elapsed={elapsed:.1f}s",
        flush=True,
    )
    if failures:
        print("[failures] first 20:")
        for index, message in failures[:20]:
            print(f"  row {index}: {message}", flush=True)


if __name__ == "__main__":
    main()