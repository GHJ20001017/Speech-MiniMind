#!/usr/bin/env python3
"""Evaluate one Route-B per-stage TTS checkpoint.

The per-stage format keeps a text-only Thinker and puts Mimi's eight codebooks in
an independent Talker decoder with 2112-class heads (real codes 0..2047,
PAD2049 and STOP2050, including STOP feedback), so HF ``generate`` cannot be used: this script runs
``forward_streams`` itself with delay-q codebook streams and independent STOPs.

Two measurements, both on dev rows of the same manifest the stage trained on:

* teacher forcing - delayed next-step accuracy / cross-entropy per codebook, i.e. how
  well the model predicts the ground-truth Mimi codes given the text. This is
  unweighted diagnostic CE (unlike training, STOP is not weighted by 10).
  It measures prediction quality but is not the logged training loss.
* free running - the model synthesises codes from text alone, Mimi decodes them
  to a waveform, and SenseVoice transcribes that back, giving a character error
  rate against the input text. The Mimi round-trip of the *ground-truth* codes is
  decoded and transcribed the same way; that is the ceiling this checkpoint could
  reach, since it is the codec's own reconstruction.

Usage::

    python scripts/evaluate_multitask_tts.py \
        --checkpoint outputs/run/stage_01_tts/model_epoch_001 \
        --data data/route_b/audio_lm_emilia_codes_24k \
        --mimi-model /gpu3/guhj/models/mimi \
        --asr-model outputs/sensevoice-small \
        --output outputs/eval_multitask_tts_epoch001 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.audio_sampling import sample_audio, GENERATION_CONFIG
from model.audio_codec import build_frozen_audio_codec  # noqa: E402
from model.audio_lm import build_vocab_spec  # noqa: E402
from trainer.train_audio_multitask import (  # noqa: E402
    AUDIO_CODEBOOK_DELAYS,
    AUDIO_STOP_ID,
    AUDIO_HEAD_SIZE,
    AUDIO_PAD_ID,
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    build_multitask_batch,
    load_multitask_checkpoint,
)

CHANCE_CE = float(np.log(AUDIO_HEAD_SIZE))


def load_rows(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def code_path(row: dict, manifest: Path) -> Path:
    path = Path(row["codes"])
    return path if path.is_absolute() else manifest.parent / path


def parquet_codec_metadata(mimi_model: str | None) -> dict:
    """The upstream format declares codec geometry, not encoder identity."""
    if not mimi_model or not mimi_model.strip():
        raise ValueError("Parquet has no codec metadata; require explicit --mimi-model")
    return {
        "codec_type": "mimi", "codec_model": mimi_model,
        "num_codebooks": NUM_CODEBOOKS, "codebook_size": CODEBOOK_SIZE,
        "sample_rate": 24000, "input_sample_rate": 24000, "frame_rate_hz": 12.5,
        "metadata_source": "upstream-declared MiniMind-O Mimi8/2048/24k/12.5Hz",
        "original_encoder_identity_verified": False,
        "codec_model_source": "explicit user selection, not verified original encoder",
    }


def select_tts_rows(data: Path, split: str = "dev", seed: int = 7, limit: int = 0):
    """Share training's parquet parsing/split; bound materialized audio samples.

    Returned count describes the selected prefix, never the complete split.
    JSONL cache paths and relative code resolution retain their old semantics.
    """
    if limit < 0:
        raise ValueError("selection limit must be nonnegative")
    if data.suffix.lower() == ".parquet":
        from dataset.minimind_t2a_dataset import MiniMindT2ADataset
        split_name = split.removesuffix(".jsonl")
        dataset = MiniMindT2ADataset(data, task="tts", split=split_name, seed=seed, limit=limit)
        rows = []
        for index, metadata in enumerate(dataset.rows):
            sample = dataset[index]
            rows.append(dict(metadata, text=sample.text, codes=sample.codes))
        return rows, data
    manifest = data / (split if split.endswith(".jsonl") else f"{split}.jsonl")
    rows = load_rows(manifest)
    return (rows[:limit] if limit else rows), manifest


def load_codes(row: dict, manifest: Path) -> torch.Tensor:
    if isinstance(row["codes"], torch.Tensor):
        return row["codes"].long()
    array = np.load(code_path(row, manifest), allow_pickle=False)
    if array.ndim == 1:
        array = array[None, :]
    return torch.from_numpy(np.asarray(array, dtype=np.int64))


def normalize(text: str) -> str:
    """Keep CJK/alphanumerics only, so punctuation cannot inflate the error rate.

    SenseVoice returns its control markers inline (``<|zh|><|NEUTRAL|>`` ...);
    those are metadata, not recognised speech, so they are dropped before scoring.
    """
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return "".join(ch.lower() for ch in text if ch.isalnum())


def edit_distance(reference: str, hypothesis: str) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref != hyp),
            ))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def save_wav(codec, codes: torch.Tensor, path: Path) -> float:
    """Decode ``(Q, T)`` codes and write a 24 kHz wav; return its duration."""
    if codes.dim() != 2 or codes.size(1) == 0:
        return 0.0
    codes = codes.long().contiguous().cpu()
    wav, lengths = codec.decode(codes.unsqueeze(0), torch.tensor([codes.size(1)]))
    samples = int(lengths[0])
    sf.write(path, wav[0, :samples].numpy(), codec.sample_rate, subtype="PCM_16")
    return samples / codec.sample_rate


@torch.inference_mode()
def teacher_forced(model, tokenizer, spec, codes: torch.Tensor, text: str, device):
    """Diagnostic delayed next-step accuracy / unweighted CE, including STOP."""
    sample = type("S", (), {"task": "tts", "codes": codes, "text": text, "prompt_codes": None})()
    inputs, _, audio_targets, audio_inputs, mask = build_multitask_batch(
        tokenizer, spec, [sample], int(model.config.max_position_embeddings)
    )
    inputs, audio_targets = inputs.to(device), audio_targets.to(device)
    audio_inputs, mask = audio_inputs.to(device), mask.to(device)
    streams = model.forward_streams(
        inputs, audio_inputs, attention_mask=mask, use_cache=False
    )
    hidden = streams.audio_hidden
    # Loss at position t predicts the label at t+1, matching multitask_loss.
    flat = hidden[:, :-1].reshape(-1, hidden.size(-1))
    labels = audio_targets[:, 1:].reshape(-1, NUM_CODEBOOKS)
    per_head, stop_count, stop_hits, total = [], 0, 0, 0
    for q, head in enumerate(model.audio_streams.heads):
        selected = labels[:, q] != -1
        count = int(selected.sum())
        if not count:
            per_head.append({"codebook": q, "frames": 0})
            continue
        target = labels[selected, q]
        logits = head(flat[selected]).float()
        ce = float(F.cross_entropy(logits, target).item())
        hit = int((logits.argmax(dim=-1) == target).sum().item())
        stop = target == AUDIO_STOP_ID
        stop_frames = int(stop.sum().item())
        stop_count += stop_frames
        stop_hits += int((logits.argmax(dim=-1)[stop] == AUDIO_STOP_ID).sum().item())
        total += count
        per_head.append({
            "codebook": q, "frames": count, "ce": ce, "ppl": float(np.exp(ce)),
            "accuracy": hit / count,
        })
    return {
        "audio_frames_scored": total,
        "mean_ce": float(np.mean([h["ce"] for h in per_head if h["frames"]])),
        "mean_accuracy": float(np.mean([h["accuracy"] for h in per_head if h["frames"]])),
        "stop_tokens": stop_count,
        "stop_accuracy": (stop_hits / stop_count) if stop_count else None,
        "per_codebook": per_head,
    }


@torch.inference_mode()
def synthesize(model, tokenizer, spec, text: str, max_frames: int, device):
    """Decode delay-q streams, then undo diagonals into complete ``(Q, T)`` frames.

    Each stream stops independently. The returned history counts streams that
    have emitted STOP cumulatively, not simultaneous votes. Inconsistent stop
    lengths are cropped to their common complete prefix; STOP is never decoded.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prefix += list(tokenizer(text, add_special_tokens=False)["input_ids"])
    prefix += [spec.audio_bos_id]
    delay = max(AUDIO_CODEBOOK_DELAYS)
    # Match the training sequence: prefix + frames + delay + final STOP row.
    available = int(model.config.max_position_embeddings) - len(prefix) - delay - 1
    if available < 1:
        raise ValueError("TTS text leaves no room for audio, codebook delays and stop in model context")
    if max_frames > available:
        raise ValueError("requested TTS frames exceed model context capacity; refusing to truncate")
    inputs = torch.tensor([prefix], dtype=torch.long, device=device)
    audio_inputs = torch.full(
        (1, inputs.size(1), NUM_CODEBOOKS), AUDIO_PAD_ID, dtype=torch.long, device=device
    )
    attention = torch.ones_like(inputs)
    out = model.forward_streams(
        inputs, audio_inputs, attention_mask=attention, use_cache=True
    )
    past, hidden = out.past_key_values, out.audio_hidden
    streams = [[] for _ in range(NUM_CODEBOOKS)]
    sampled = [[] for _ in range(NUM_CODEBOOKS)]
    stopped = [False] * NUM_CODEBOOKS
    vote_history = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for step in range(max_frames + delay + 1):
        feedback = [AUDIO_PAD_ID] * NUM_CODEBOOKS
        for q, head in enumerate(model.audio_streams.heads):
            frame = step - AUDIO_CODEBOOK_DELAYS[q]
            if frame < 0:
                sampled[q].append(AUDIO_PAD_ID)
                continue
            top = sample_audio(head(hidden[:, -1])[0], sampled[q])
            sampled[q].append(top)
            feedback[q] = top
            if top >= CODEBOOK_SIZE:
                stopped[q] = True
            elif not stopped[q] and frame < max_frames:
                streams[q].append(top)
                feedback[q] = top
        vote_history.append(sum(stopped))
        if all(stopped) or step == max_frames + delay:
            break
        next_codes = torch.tensor([[feedback]], dtype=torch.long, device=device)
        # Started streams keep sampled feedback, including all special IDs.
        next_text = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
        out = model.forward_streams(
            next_text, next_codes, attention_mask=attention,
            past_key_values=past, use_cache=True,
        )
        past, hidden = out.past_key_values, out.audio_hidden
    frames = min(map(len, streams))
    codes = torch.tensor([stream[:frames] for stream in streams], dtype=torch.long)
    return codes, vote_history


def build_transcriber(model_dir: str | None, device: torch.device):
    if not model_dir:
        return None
    try:
        from funasr import AutoModel
    except ImportError:
        print("WARNING: funasr not installed; skipping ASR round-trip", flush=True)
        return None
    model = AutoModel(model=model_dir, device=str(device), disable_update=True, disable_pbar=True)
    return lambda wav_path: model.generate(
        input=str(wav_path), cache={}, language="auto", use_itn=True, batch_size_s=60,
    )[0]["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="JSONL cache directory or MiniMind-O .parquet")
    parser.add_argument("--split", default="dev.jsonl")
    parser.add_argument("--mimi-model", help="required explicit decoder for parquet; JSONL keeps its legacy default")
    parser.add_argument("--asr-model", default=None, help="SenseVoice dir; omit to skip ASR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-forcing-rows", "--codebook-samples", type=int, default=64)
    parser.add_argument("--synthesis-rows", "--num-samples", type=int, default=6)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-frames-filter", type=int, default=250)
    parser.add_argument("--max-new-frames", type=int, default=400)
    args = parser.parse_args()

    if min(args.teacher_forcing_rows, args.synthesis_rows) < 0 or max(args.teacher_forcing_rows, args.synthesis_rows) == 0:
        parser.error("sample counts must be nonnegative and at least one must be positive")
    is_parquet = args.data.suffix.lower() == ".parquet"
    codec_metadata = parquet_codec_metadata(args.mimi_model) if is_parquet else None
    if args.mimi_model is None:
        args.mimi_model = "/gpu3/guhj/models/mimi"
    stage = json.loads((args.checkpoint.parent / "config.json").read_text()) if is_parquet else {}
    seed = int(stage.get("seed", 7))
    device = torch.device(args.device)
    torch.manual_seed(seed)
    args.output.mkdir(parents=True, exist_ok=True)
    # Parquet's limit bounds decoding/materialization; frame filtering may reduce
    # the actual scored count below the request, which the report makes explicit.
    selection_limit = max(args.teacher_forcing_rows, args.synthesis_rows) if is_parquet else 0
    rows, manifest = select_tts_rows(args.data, args.split, seed, selection_limit)

    model, tokenizer = load_multitask_checkpoint(args.checkpoint, device)
    spec = build_vocab_spec(tokenizer, CODEBOOK_SIZE, NUM_CODEBOOKS)
    codec = build_frozen_audio_codec(
        "mimi", model_id=args.mimi_model, device=device, num_codebooks=NUM_CODEBOOKS
    )
    if codec_metadata is not None:
        for key in ("num_codebooks", "codebook_size", "sample_rate", "frame_rate_hz"):
            if getattr(codec, key) != codec_metadata[key]:
                raise ValueError(f"Loaded Mimi {key} differs from upstream-declared parquet geometry")
    transcriber = build_transcriber(args.asr_model, device)
    print(f"checkpoint={args.checkpoint} selected_rows={len(rows)} chance_ce={CHANCE_CE:.4f}", flush=True)

    # Teacher forcing over a stable slice of the dev split.
    scored, teacher_rows = [], []
    for row in rows:
        if len(teacher_rows) >= args.teacher_forcing_rows:
            break
        codes = load_codes(row, manifest)
        if codes.size(0) != NUM_CODEBOOKS or not (args.min_frames <= codes.size(1) <= args.max_frames_filter):
            continue
        if codes.min() < 0 or codes.max() >= CODEBOOK_SIZE:
            continue
        teacher_rows.append(row)
    for index, row in enumerate(teacher_rows):
        codes = load_codes(row, manifest)
        metrics = teacher_forced(model, tokenizer, spec, codes, row.get("text", ""), device)
        metrics["row"] = index
        metrics["frames"] = int(codes.size(1))
        scored.append(metrics)
    if scored:
        summary = {
            "rows": len(scored),
            "mean_ce": float(np.mean([s["mean_ce"] for s in scored])),
            "mean_accuracy": float(np.mean([s["mean_accuracy"] for s in scored])),
            "stop_accuracy": float(np.mean([s["stop_accuracy"] for s in scored if s["stop_accuracy"] is not None])),
            "per_codebook_ce": [
                float(np.mean([s["per_codebook"][q]["ce"] for s in scored]))
                for q in range(NUM_CODEBOOKS)
            ],
            "per_codebook_accuracy": [
                float(np.mean([s["per_codebook"][q]["accuracy"] for s in scored]))
                for q in range(NUM_CODEBOOKS)
            ],
        }
        print("TEACHER_FORCED " + json.dumps(summary), flush=True)

    # Free-running synthesis plus the codec ceiling on the same texts.
    synthesis = []
    for index, row in enumerate(rows[: args.synthesis_rows]):
        codes = load_codes(row, manifest)
        if codes.size(0) != NUM_CODEBOOKS:
            continue
        text = row.get("text", "")
        sample_dir = args.output / f"sample_{index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        # The ceiling is the codec's own reconstruction, so the reference keeps
        # every frame: clipping it would score a truncated utterance against the
        # full transcript and inflate its CER.
        reference_seconds = save_wav(codec, codes, sample_dir / "reference.wav")
        generated, votes = synthesize(
            model, tokenizer, spec, text, args.max_new_frames, device
        )
        generated_seconds = save_wav(codec, generated, sample_dir / "generated.wav")
        record = {
            "row": index, "text": text, "reference_frames": int(codes.size(1)),
            "generated_frames": int(generated.size(1)),
            "generated_seconds": generated_seconds,
            "reference_seconds": reference_seconds,
            "distinct_codes": int(torch.unique(generated).numel()) if generated.numel() else 0,
            "stop_voted": bool(votes and votes[-1] == NUM_CODEBOOKS),
        }
        if transcriber is not None:
            reference_text = transcriber(sample_dir / "reference.wav")
            generated_text = transcriber(sample_dir / "generated.wav") if generated.size(1) else ""
            record.update({
                "reference_asr": reference_text,
                "generated_asr": generated_text,
                "reference_cer": cer(text, reference_text),
                "generated_cer": cer(text, generated_text),
            })
        synthesis.append(record)
        print("SYNTHESIS " + json.dumps(record, ensure_ascii=False), flush=True)

    if synthesis:
        with_transcript = [s for s in synthesis if "generated_cer" in s]
        if with_transcript:
            print("FREE_RUNNING " + json.dumps({
                "samples": len(with_transcript),
                "mean_generated_cer": float(np.mean([s["generated_cer"] for s in with_transcript])),
                "mean_reference_cer": float(np.mean([s["reference_cer"] for s in with_transcript])),
                "mean_generated_frames": float(np.mean([s["generated_frames"] for s in with_transcript])),
            }), flush=True)

    (args.output / "metrics.json").write_text(json.dumps({
        "generation": GENERATION_CONFIG, "seed": seed,
        "checkpoint": str(args.checkpoint), "split": args.split.removesuffix(".jsonl"), "device": str(device),
        "data": str(args.data), "data_source": "minimind_t2a_parquet" if is_parquet else "jsonl_cache",
        "seed": seed, "codec_metadata": codec_metadata,
        "selection": {"loaded_rows": len(rows), "limit": selection_limit,
                      "complete_split_count_known": not is_parquet,
                      "teacher_requested": args.teacher_forcing_rows, "teacher_scored": len(scored),
                      "synthesis_requested": args.synthesis_rows, "synthesis_scored": len(synthesis)},
        "chance_ce": CHANCE_CE, "teacher_forced": scored, "synthesis": synthesis,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
