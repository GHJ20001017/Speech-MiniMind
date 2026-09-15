"""M0 gate: measure a frozen codec's reconstruction quality on our own audio.

Before committing to a codec for Route B we must know how much information is
lost by the waveform -> codes -> waveform round trip.  This script reports:

* **mel distortion** - mean absolute error between log-mel spectrograms of the
  original and the reconstruction (always available, needs no extra packages);
* **STOI / PESQ** - if ``pystoi`` / ``pesq`` happen to be installed;
* **round-trip ASR CER** - WER/CER of the reconstructed audio transcribed by the
  Route-A SenseVoice frontend, which is the metric that actually matters for
  "is the generated speech intelligible".

It also dumps paired ``orig_*.wav`` / ``recon_*.wav`` files so you can listen.

Usage::

    python scripts/eval_codec_reconstruction.py \
        --data data/aishell1/processed --split dev --num 20 \
        --codec-type mimi --device cuda:0 \
        --output outputs/05_route_b_codec_check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from model.audio_codec import build_frozen_audio_codec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="AISHELL processed dir (CSV) or a dir of JSONL manifests")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num", type=int, default=20, help="how many utterances to check")
    parser.add_argument("--codec-type", default="mimi", choices=("mimi", "encodec"))
    parser.add_argument("--codec-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/05_route_b_codec_check"))
    parser.add_argument("--save-audio", type=int, default=10,
                        help="how many orig/recon WAV pairs to write for listening")
    parser.add_argument("--asr-cer", action=argparse.BooleanOptionalAction, default=True,
                        help="compute round-trip ASR CER with the SenseVoice frontend")
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_aishell_rows(data: Path, split: str, num: int) -> list[tuple[str, str]]:
    """Return ``[(audio_path, transcript), ...]`` from an AISHELL CSV manifest."""
    manifest = data / f"{split}.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    rows: list[tuple[str, str]] = []
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            if not path.is_absolute():
                path = data / path
            rows.append((str(path), row.get("text", "")))
    return rows[:num] if num else rows


def log_mel_np(audio: np.ndarray, sample_rate: int, n_mels: int = 80) -> np.ndarray:
    from scripts.analyze_audio import log_mel

    features, _, _, _ = log_mel(audio, sample_rate, 25, 10, n_mels)
    return features


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate via Levenshtein distance (no external deps)."""
    reference = "".join(reference.split())
    hypothesis = "".join(hypothesis.split())
    if not reference:
        return float("nan")
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1] / len(reference)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    codec = build_frozen_audio_codec(args.codec_type, args.codec_model, args.device)
    print(f"codec: {args.codec_type} Q={codec.num_codebooks} vocab={codec.codebook_size} "
          f"sr={codec.sample_rate} frame_rate={codec.frame_rate_hz:.2f}Hz")

    rows = load_aishell_rows(args.data, args.split, args.num)
    if not rows:
        raise SystemExit("no utterances selected")

    asr = None
    if args.asr_cer:
        try:
            from funasr import AutoModel

            asr = AutoModel(model=args.sensevoice_model, device=args.device,
                            disable_update=True)
            print("round-trip ASR enabled")
        except Exception as error:  # noqa: BLE001 - optional dependency
            print(f"round-trip ASR disabled ({error})")

    import soundfile as sf

    base_samples: list[dict] = []
    for index, (path, transcript) in enumerate(rows):
        waveform, rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        base_samples.append({"index": index, "path": path, "transcript": transcript,
                             "waveform": waveform.astype(np.float32), "rate": int(rate)})

    records: list[dict] = []
    for item in base_samples:
        index, path, transcript = item["index"], item["path"], item["transcript"]
        waveform, rate = item["waveform"], item["rate"]
        tensor = torch.from_numpy(waveform)[None, :]
        lengths = torch.tensor([tensor.size(1)], dtype=torch.long)

        codes, code_lengths = codec.encode(tensor, lengths, rate)
        recon, recon_lengths = codec.decode(codes, code_lengths)
        recon_np = recon[0, : int(recon_lengths[0])].numpy()

        mel_orig = log_mel_np(waveform, rate)
        mel_recon = log_mel_np(recon_np, codec.sample_rate)
        frames = min(len(mel_orig), len(mel_recon))
        mel_mae = float(np.mean(np.abs(mel_orig[:frames] - mel_recon[:frames])))

        record = {
            "index": index,
            "path": path,
            "duration_s": round(len(waveform) / rate, 3),
            "frames": int(code_lengths[0]),
            "tokens_per_s": round(float(code_lengths[0]) / max(len(waveform) / rate, 1e-6), 3),
            "mel_mae": round(mel_mae, 4),
        }

        try:
            import pystoi

            record["stoi"] = round(float(pystoi.stoi(waveform, recon_np, rate)), 4)
        except ImportError:
            pass
        try:
            from pesq import pesq

            record["pesq"] = round(float(pesq(rate, waveform, recon_np, "wb")), 4)
        except Exception:  # noqa: BLE001 - optional dependency
            pass

        if index < args.save_audio:
            sf.write(str(audio_dir / f"{index:03d}_orig.wav"), waveform, rate, subtype="PCM_16")
            sf.write(str(audio_dir / f"{index:03d}_recon.wav"), recon_np, codec.sample_rate,
                     subtype="PCM_16")
        item["recon"] = recon_np

        records.append(record)
        print(f"[{index + 1}/{len(rows)}] frames={record['frames']} mel_mae={record['mel_mae']}",
              flush=True)

    # Round-trip ASR: transcribe every reconstruction, then score the CERs.
    if asr is not None:
        try:
            import re

            for position, item in enumerate(base_samples):
                result = asr.generate(input=item["recon"], cache={}, language="zh",
                                      use_itn=False, batch_size_s=60)
                hypothesis = result[0]["text"] if result else ""
                hypothesis = re.sub(r"<\|[^|]*\|>", "", hypothesis).strip()
                records[position]["asr_hypothesis"] = hypothesis
                if item["transcript"]:
                    records[position]["cer"] = round(cer(item["transcript"], hypothesis), 4)
        except Exception as error:  # noqa: BLE001 - optional dependency
            print(f"round-trip ASR failed ({error})")

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, float] = {
        "utterances": len(records),
        "mel_mae_mean": float(np.mean([r["mel_mae"] for r in records])),
        "tokens_per_s_mean": float(np.mean([r["tokens_per_s"] for r in records])),
    }
    for key in ("stoi", "pesq", "cer"):
        values = [r[key] for r in records if key in r]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    (args.output / "summary.json").write_text(
        json.dumps(summary | {
            "codec_type": args.codec_type,
            "codec_model": args.codec_model,
            "num_codebooks": codec.num_codebooks,
            "codebook_size": codec.codebook_size,
            "sample_rate": codec.sample_rate,
            "frame_rate_hz": codec.frame_rate_hz,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"listened pairs: {args.save_audio} -> {audio_dir}")


if __name__ == "__main__":
    main()
