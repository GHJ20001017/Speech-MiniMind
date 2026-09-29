#!/usr/bin/env python3
"""Text-only TTS / dev listening samples for train_audio_multitask.py checkpoints.

Read the tokenizer/backbone/audio heads from the checkpoint, the data path from
its parent stage config, and Mimi settings from the training cache metadata.
No chat template, reference audio conditioning, or HF text generate is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.audio_codec import build_frozen_audio_codec
from model.audio_lm import build_vocab_spec
from model.audio_sampling import GENERATION_CONFIG
from scripts.evaluate_multitask_tts import (
    build_transcriber, cer, load_codes, parquet_codec_metadata, select_tts_rows, save_wav, synthesize,
)
from trainer.train_audio_multitask import (
    AUDIO_CODEBOOK_DELAYS, CODEBOOK_SIZE, NUM_CODEBOOKS,
    load_multitask_checkpoint, validate_init_checkpoint,
)


def resolve_config(checkpoint: Path, data: Path | None, mimi_model: str | None):
    metadata = validate_init_checkpoint(checkpoint)
    if metadata.get("task") != "tts":
        raise ValueError("Use the trained TTS checkpoint, not the ASR initialization checkpoint")
    stage = json.loads((checkpoint.parent / "config.json").read_text())
    if stage.get("task") != "tts":
        raise ValueError("Stage config is not a TTS training config")
    data = data or Path(stage["data"])
    if data.suffix.lower() == ".parquet":
        return data, mimi_model, stage, parquet_codec_metadata(mimi_model)
    cache = json.loads((data / "metadata.json").read_text())
    expected = {
        "codec_type": "mimi", "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE, "sample_rate": 24000,
        "input_sample_rate": 24000, "frame_rate_hz": 12.5,
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f"Training cache mismatch: {key}={cache.get(key)!r}, expected {value!r}")
    trained_codec = cache.get("codec_model") or "kyutai/mimi"
    if mimi_model is not None and mimi_model != trained_codec:
        raise ValueError(f"--mimi-model must match training cache codec_model={trained_codec!r}")
    return data, trained_codec, stage, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite old audio")
    parser.add_argument("--data", type=Path, help="cache directory or MiniMind-O parquet if relocated; defaults to stage config")
    parser.add_argument("--mimi-model", help="required decoder for parquet; optional strict codec assertion for JSONL cache")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--text", action="append", help="repeat for multiple texts; otherwise use dev rows")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-new-frames", type=int, default=500, help="safety cap, 500 frames = 40s; not a target duration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-model", help="optional SenseVoice directory for round-trip CER")
    args = parser.parse_args()
    if args.num_samples < 1 or args.max_new_frames < 1:
        parser.error("num-samples and max-new-frames must be positive")
    if args.text is not None and any(not text.strip() for text in args.text):
        parser.error("--text must not be blank")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    data, mimi_model, stage, cache = resolve_config(args.checkpoint, args.data, args.mimi_model)
    seed = int(stage.get("seed", 7))
    if args.text is not None:
        rows, manifest = [{"text": text} for text in args.text], data
    else:
        rows, manifest = select_tts_rows(data, args.split, seed, args.num_samples)
    if not rows or any(not row.get("text", "").strip() for row in rows):
        raise ValueError("No nonempty TTS text in selected rows")
    device = torch.device(args.device)
    torch.manual_seed(int(stage.get("seed", 7)))
    model, tokenizer = load_multitask_checkpoint(args.checkpoint, device)
    spec = build_vocab_spec(tokenizer, CODEBOOK_SIZE, NUM_CODEBOOKS)
    codec = build_frozen_audio_codec("mimi", model_id=mimi_model, device=device, num_codebooks=NUM_CODEBOOKS)
    for key in ("num_codebooks", "codebook_size", "sample_rate", "frame_rate_hz"):
        if getattr(codec, key) != cache[key]:
            raise ValueError(f"Loaded Mimi {key} differs from training cache")
    transcriber = build_transcriber(args.asr_model, device)
    if args.asr_model and transcriber is None:
        raise RuntimeError("ASR requested but unavailable; install funasr or omit --asr-model")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "checkpoint": str(args.checkpoint), "training_config": stage,
        "cache_metadata": cache, "data": str(data), "mimi_model": mimi_model,
        "device": str(device), "decoding": "sampled delay-q; first special stops retained stream; common complete frames",
        "generation": GENERATION_CONFIG,
        "data_source": "explicit_text" if args.text is not None else (
            "minimind_t2a_parquet" if data.suffix.lower() == ".parquet" else "jsonl_cache"),
        "split": None if args.text is not None else args.split, "seed": seed,
        "selection": {"selected_rows": len(rows), "requested_rows": len(args.text) if args.text is not None else args.num_samples},
        "audio_codebook_delays": list(AUDIO_CODEBOOK_DELAYS),
        "max_new_frames": args.max_new_frames, "samples": [],
    }
    for index, row in enumerate(rows):
        text = row["text"]
        sample_dir = args.output / f"sample_{index:02d}"
        sample_dir.mkdir()
        (sample_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
        generated, votes = synthesize(model, tokenizer, spec, text, args.max_new_frames, device)
        stopped = bool(votes and votes[-1] == NUM_CODEBOOKS)
        record = {
            "sample": sample_dir.name, "text": text,
            "generated_frames": generated.size(1), "stop_votes": votes,
            "stop_reason": ("audio_stop" if stopped else "max_new_frames"),
            "generated_wav": None,
        }
        np.save(sample_dir / "generated_codes.npy", generated.numpy())
        record["generated_seconds"] = save_wav(codec, generated, sample_dir / "generated.wav")
        if generated.size(1):
            record["generated_wav"] = str(sample_dir / "generated.wav")
        else:
            record["warning"] = "Immediate stop: no speech produced"
        if "codes" in row:
            reference = load_codes(row, manifest)
            if reference.ndim != 2 or reference.size(0) != NUM_CODEBOOKS or not reference.size(1):
                raise ValueError("Invalid reference code shape")
            if (reference < 0).any() or (reference >= CODEBOOK_SIZE).any():
                raise ValueError("Reference codes outside Mimi vocabulary")
            record["reference_seconds"] = save_wav(codec, reference, sample_dir / "reference.wav")
            if transcriber:
                reference_text = transcriber(sample_dir / "reference.wav")
                record.update(reference_asr=reference_text, reference_cer=cer(text, reference_text))
        if transcriber:
            generated_text = transcriber(sample_dir / "generated.wav") if generated.size(1) else ""
            record.update(generated_asr=generated_text, generated_cer=cer(text, generated_text))
        report["samples"].append(record)
        (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(f"Listen to {args.output}/sample_*/generated.wav; reference.wav is the codec reconstruction, not a voice prompt.")


if __name__ == "__main__":
    main()
