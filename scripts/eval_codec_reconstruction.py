"""M0 gate: measure a frozen codec's reconstruction quality on our own audio.

Before committing to a codec for Route B we must know how much information is
lost by the waveform -> codes -> waveform round trip.  This script reports:

* **mel distortion** - mean absolute error between log-mel spectrograms of the
  original and the reconstruction (always available, needs no extra packages);
* **STOI / PESQ** - if ``pystoi`` / ``pesq`` happen to be installed;
* **round-trip ASR CER / WER** - character (zh) or word (en) error rate of the
  reconstructed audio transcribed by the Route-A SenseVoice frontend, which is
  the metric that actually matters for "is the generated speech intelligible".

It also dumps paired ``orig_*.wav`` / ``recon_*.wav`` files so you can listen.

``--data`` accepts either an AISHELL-style processed dir (``{split}.csv`` with
``path``/``text``), or any dir of the project JSONL manifests (``{split}.jsonl``
with ``audio`` + ``answer``), or a single manifest file.  That lets the same
gate run on Chinese read speech *and* on the English/mixed corpora
(``voiceassistant400k_50k``, ``moss_speech_qa``) without converting CSV first.

Usage::

    # Chinese read speech (AISHELL CSV)
    python scripts/eval_codec_reconstruction.py \
        --data data/aishell1/processed --split dev --num 20 \
        --codec-type mimi --device cuda:0 \
        --output outputs/05_route_b_codec_check

    # English corpus (JSONL manifest), scored with WER
    python scripts/eval_codec_reconstruction.py \
        --data data/voiceassistant400k_50k --split dev --num 20 \
        --asr-language en --codec-type mimi --device cuda:0 \
        --output outputs/05_route_b_codec_check_en

    # sft_a2a answer-audio domain (codes only + answer_text), zh or en
    python scripts/eval_codec_reconstruction.py \
        --data data/route_b/s2s --split dev --num 50 \
        --asr-language en --codec-type mimi --device cuda:0 \
        --output outputs/05_route_b_codec_check_s2s_en
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
                        help="AISHELL processed dir (CSV), a dir of JSONL manifests, or one manifest")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num", type=int, default=20, help="how many utterances to check")
    parser.add_argument("--codec-type", default="mimi", choices=("mimi", "encodec"))
    parser.add_argument("--codec-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/05_route_b_codec_check"))
    parser.add_argument("--save-audio", type=int, default=10,
                        help="how many orig/recon WAV pairs to write for listening")
    parser.add_argument("--asr-cer", action=argparse.BooleanOptionalAction, default=True,
                        help="compute round-trip ASR error rate with the SenseVoice frontend")
    parser.add_argument("--asr-language", default="zh", choices=("zh", "en", "yue", "ja", "ko", "auto"),
                        help="language hint for the round-trip SenseVoice ASR")
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_aishell_rows(data: Path, split: str, num: int) -> list[dict]:
    """Return manifest rows from an AISHELL CSV manifest.

    AISHELL manifests store repo-root-relative paths (``data/aishell1/...``), so
    resolve against the repo root first and fall back to the manifest directory.
    """
    manifest = data / f"{split}.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    rows: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            if not path.is_absolute():
                candidate = ROOT / path
                path = candidate if candidate.exists() else data / path
            rows.append({"audio": str(path), "codes": None,
                         "transcript": (row.get("text") or "").strip()})
    return rows[:num] if num else rows


def load_jsonl_rows(manifest: Path, num: int) -> list[dict]:
    """Return manifest rows from a project JSONL manifest.

    Three row shapes are recognised:

    * ``{"audio", "instruction"|"answer"|"text"}`` - raw audio corpora (moss,
      VoiceAssistant-400K, AISHELL exports).  The waveform is re-encoded on the
      fly, so mel distortion / STOI / PESQ are available.
    * ``{"answer_codes", "answer_text"}`` - the Route-B s2s manifest, where the
      answer audio only exists as ``.npy`` codes.  There is no reference
      waveform, so only the round-trip error rate is scored (this is how the
      ``sft_a2a`` answer-audio domain, zh *and* en, gets validated).
    * ``{"codes"}`` - the Route-B B0 token cache; also code-only, but has no
      transcript, so nothing is scored (it exists so the loader does not crash).
    """
    rows: list[dict] = []
    base = manifest.parent

    def resolve(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        candidate = ROOT / path
        return str(candidate if candidate.exists() else base / path)

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            audio = str(record.get("audio", "")).strip()
            codes = str(record.get("answer_codes") or record.get("codes") or "").strip()
            if not audio and not codes:
                continue
            transcript = str(
                record.get("answer_text") or record.get("instruction") or record.get("text")
                or record.get("transcript") or record.get("answer") or ""
            ).strip()
            rows.append({
                "audio": resolve(audio) if audio else None,
                "codes": resolve(codes) if codes else None,
                "transcript": transcript,
            })
    return rows[:num] if num else rows


def resolve_rows(data: Path, split: str, num: int) -> tuple[list[dict], str]:
    """Pick the manifest loader from what ``--data`` actually contains.

    Returns ``(rows, kind)`` where ``kind`` is ``"csv"`` or ``"jsonl"`` so the
    caller can label the summary.
    """
    if data.is_file():
        return load_jsonl_rows(data, num), "jsonl"
    if (data / f"{split}.csv").exists():
        return load_aishell_rows(data, split, num), "csv"
    if (data / f"{split}.jsonl").exists():
        return load_jsonl_rows(data / f"{split}.jsonl", num), "jsonl"
    raise SystemExit(
        f"no manifest found under {data} (looked for {split}.csv and {split}.jsonl)"
    )


def log_mel_np(audio: np.ndarray, sample_rate: int, n_mels: int = 80) -> np.ndarray:
    from scripts.analyze_audio import log_mel

    features, _, _, _ = log_mel(audio, sample_rate, 25, 10, n_mels)
    return features


def cer(reference: str, hypothesis: str, level: str = "char") -> float:
    """Error rate via Levenshtein distance (no external deps).

    ``level="char"`` scores Chinese (character error rate); ``level="word"``
    scores space-delimited languages (word error rate, WER).  Punctuation is
    stripped for the word level so ASR punctuation differences do not inflate it.
    """
    if level == "word":
        import re

        def strip(text: str) -> list[str]:
            return [t for t in re.sub(r"[^\w\s']", " ", text.lower()).split() if t]

        reference_units: list[str] = strip(reference)
        hypothesis_units: list[str] = strip(hypothesis)
    else:
        reference_units = list("".join(reference.split()))
        hypothesis_units = list("".join(hypothesis.split()))
    if not reference_units:
        return float("nan")
    previous = list(range(len(hypothesis_units) + 1))
    for i, ref_unit in enumerate(reference_units, start=1):
        current = [i]
        for j, hyp_unit in enumerate(hypothesis_units, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_unit != hyp_unit),
            ))
        previous = current
    return previous[-1] / len(reference_units)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    codec = build_frozen_audio_codec(args.codec_type, args.codec_model, args.device)
    print(f"codec: {args.codec_type} Q={codec.num_codebooks} vocab={codec.codebook_size} "
          f"sr={codec.sample_rate} frame_rate={codec.frame_rate_hz:.2f}Hz")

    rows, manifest_kind = resolve_rows(args.data, args.split, args.num)
    if not rows:
        raise SystemExit("no utterances selected")
    transcript_level = "word" if args.asr_language == "en" else "char"

    asr = None
    if args.asr_cer:
        try:
            from funasr import AutoModel

            asr = AutoModel(model=args.sensevoice_model, device=args.device,
                            disable_update=True)
            print(f"round-trip ASR enabled (language={args.asr_language}, "
                  f"scoring={'WER' if transcript_level == 'word' else 'CER'})")
        except Exception as error:  # noqa: BLE001 - optional dependency
            print(f"round-trip ASR disabled ({error})")

    import soundfile as sf

    base_samples: list[dict] = []
    for index, row in enumerate(rows):
        if row["audio"]:
            waveform, rate = sf.read(row["audio"], dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            base_samples.append({"index": index, "path": row["audio"],
                                 "transcript": row["transcript"],
                                 "waveform": waveform.astype(np.float32),
                                 "rate": int(rate), "codes": None})
        else:
            # Code-only row (Route-B answer shard): keep the codes, rebuild the
            # reference by decoding them, so the round-trip error rate still
            # measures "can a listener recover the words from these codes".
            base_samples.append({"index": index, "path": row["codes"],
                                 "transcript": row["transcript"],
                                 "waveform": None, "rate": codec.sample_rate,
                                 "codes": row["codes"]})

    records: list[dict] = []
    for item in base_samples:
        index, path = item["index"], item["path"]
        codes_path = item["codes"]

        if codes_path:
            shard = np.load(codes_path)
            if shard.ndim == 1:
                shard = shard[None, :]
            code_tensor = torch.from_numpy(np.asarray(shard, dtype=np.int64))[None, :, :]
            code_lengths = torch.tensor([code_tensor.size(2)], dtype=torch.long)
            recon, recon_lengths = codec.decode(code_tensor, code_lengths)
            recon_np = recon[0, : int(recon_lengths[0])].numpy()
            rate = codec.sample_rate
            record = {
                "index": index,
                "path": path,
                "source": "codes",
                "duration_s": round(len(recon_np) / codec.sample_rate, 3),
                "frames": int(code_lengths[0]),
                "tokens_per_s": round(float(code_lengths[0]) / max(len(recon_np) / codec.sample_rate, 1e-6), 3),
            }
            if index < args.save_audio:
                sf.write(str(audio_dir / f"{index:03d}_recon.wav"), recon_np, codec.sample_rate,
                         subtype="PCM_16")
            item["recon"] = recon_np
            records.append(record)
            print(f"[{index + 1}/{len(rows)}] codes frames={record['frames']} "
                  f"(no reference waveform)", flush=True)
            continue

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
            "source": "waveform",
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

    # Round-trip ASR: transcribe every reconstruction, then score the error rates.
    if asr is not None:
        try:
            import re

            for position, item in enumerate(base_samples):
                result = asr.generate(input=item["recon"], cache={},
                                      language=args.asr_language,
                                      use_itn=False, batch_size_s=60)
                hypothesis = result[0]["text"] if result else ""
                hypothesis = re.sub(r"<\|[^|]*\|>", "", hypothesis).strip()
                records[position]["asr_hypothesis"] = hypothesis
                if item["transcript"]:
                    records[position]["cer"] = round(
                        cer(item["transcript"], hypothesis, transcript_level), 4
                    )
        except Exception as error:  # noqa: BLE001 - optional dependency
            print(f"round-trip ASR failed ({error})")

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    mel_values = [r["mel_mae"] for r in records if "mel_mae" in r]
    summary: dict[str, float | str | int] = {
        "utterances": len(records),
        "waveform_rows": sum(1 for r in records if r.get("source") == "waveform"),
        "codes_rows": sum(1 for r in records if r.get("source") == "codes"),
        "manifest_kind": manifest_kind,
        "asr_language": args.asr_language,
        "tokens_per_s_mean": float(np.mean([r["tokens_per_s"] for r in records])),
    }
    if mel_values:
        summary["mel_mae_mean"] = float(np.mean(mel_values))
    for key in ("stoi", "pesq", "cer"):
        values = [r[key] for r in records if key in r]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    if "cer_mean" in summary:
        summary["error_rate_mean"] = summary["cer_mean"]
        summary["error_rate_level"] = transcript_level
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
