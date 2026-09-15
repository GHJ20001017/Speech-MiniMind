"""Pre-encode audio into discrete codebook shards for Route-B training.

Running the frozen codec inside the training loop is far too expensive: Mimi
encoding costs more than a full LLM step, and it would be repeated every epoch.
This script walks a manifest, batches the audio through the codec once, and
writes ``.npy`` shards plus a rewritten manifest whose rows point at the shards.

Input manifests (any split produced upstream):

* ``{"audio": ...}``                        -> pure audio LM rows (B0)
* ``{"prompt_audio","answer_audio", ...}``  -> speech-to-speech rows (B1/B2)

Output layout::

    <output>/codes/<split>/<index>[_p|_a].npy
    <output>/{train,dev}.jsonl                 # rewritten, code-path rows
    <output>/metadata.json                     # codec + counts (for cache keys)

Usage::

    python scripts/cache_audio_tokens.py \
        --data data/route_b/audio_lm --output data/route_b/audio_lm_codes \
        --codec-type mimi --device cuda:0 --batch-size 16

The cache is keyed by ``(<audio manifest>, codec, num_codebooks)`` in
``metadata.json``; re-running with a different codec writes to a different
``--output`` so stale codes are never silently reused.
"""

from __future__ import annotations

import argparse
import json
import time
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
                        help="dir with {train,dev}.jsonl, or a single JSONL")
    parser.add_argument("--dev-file", type=Path, default=None,
                        help="dev JSONL when --data is a single file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-type", default="mimi", choices=("mimi", "encodec"))
    parser.add_argument("--codec-model", default=None)
    parser.add_argument("--num-codebooks", type=int, default=None,
                        help="truncate the codec to this many codebooks (encodec)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=40.0,
                        help="skip audio longer than this (keeps shards bounded)")
    parser.add_argument("--limit", type=int, default=0, help="only N rows per split")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode even if the shard already exists")
    return parser.parse_args()


def read_audio(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit("soundfile is required. Run: python -m pip install soundfile") from error
    if not path.exists():
        return None
    try:
        waveform, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - per-file tolerance
        return None
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        return None
    return waveform.astype(np.float32), int(rate)


def resolve(manifest: Path, value: str) -> Path:
    """Resolve an audio path from a manifest row to an absolute path.

    Upstream manifests mix two conventions (repo-root-relative from AISHELL CSVs,
    manifest-relative from the moss/voiceassistant JSONLs), and some - like the
    ``audio_lm`` corpus - store absolute paths.  Try all three so a relative
    entry is never silently joined against the wrong directory.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[1]
    for candidate in (root / path, manifest.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (manifest.parent / path).resolve()


def flush_batch(codec, batch: list[dict], code_dir: Path, counters: dict) -> set[str]:
    """Encode one batch of ``{"key", "waveform", "rate"}`` items and save shards.

    Returns the set of keys that actually produced a shard, so the caller can
    drop manifest rows whose encode failed instead of pointing at a missing file.
    """
    if not batch:
        return set()
    waveforms = [torch.from_numpy(item["waveform"]) for item in batch]
    lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    rates = {item["rate"] for item in batch}
    rate = rates.pop() if len(rates) == 1 else 16000
    codes, code_lengths = codec.encode(padded, lengths, rate)
    written: set[str] = set()
    for item, code, length in zip(batch, codes, code_lengths.tolist()):
        if length <= 0:
            counters["failed"] += 1
            continue
        np.save(code_dir / f"{item['key']}.npy", code[:, :length].numpy().astype(np.int16))
        counters["encoded"] += 1
        written.add(item["key"])
    batch.clear()
    return written


def process_split(
    args: argparse.Namespace,
    codec,
    manifest: Path,
    split: str,
    output: Path,
) -> None:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    code_dir = output / "codes" / split
    code_dir.mkdir(parents=True, exist_ok=True)

    def rel_path(key: str) -> str:
        return f"codes/{split}/{key}.npy"

    out_rows: list[dict] = []
    counters = {"encoded": 0, "failed": 0, "skipped_long": 0, "reused": 0}
    batch: list[dict] = []
    started = time.time()

    for index, row in enumerate(rows):
        is_s2s = "answer_audio" in row or "prompt_audio" in row
        # ``pieces`` maps a source field to either the shard it will live in, or
        # ``None`` when the audio could not be read/skipped.  Encoding itself is
        # deferred (batching), so we must NOT probe the filesystem here: the
        # shard is only written when the batch flushes.
        pieces: dict[str, str | None] = {}

        for field, suffix in (("prompt_audio", "p"), ("answer_audio", "a"), ("audio", "")):
            value = row.get(field)
            if not value:
                continue
            key = f"{index:07d}_{suffix}" if suffix else f"{index:07d}"
            target = code_dir / f"{key}.npy"
            if target.exists() and not args.overwrite:
                pieces[field] = rel_path(key)
                counters["reused"] += 1
                continue
            loaded = read_audio(resolve(manifest, value))
            if loaded is None:
                pieces[field] = None
                continue
            waveform, rate = loaded
            if waveform.size / rate > args.max_seconds:
                counters["skipped_long"] += 1
                pieces[field] = None
                continue
            # Queue for encoding and record where it will land; the shard is
            # written on the next flush, so do not look for it on disk yet.
            batch.append({"key": key, "waveform": waveform, "rate": rate})
            pieces[field] = rel_path(key)

        if len(batch) >= args.batch_size:
            flush_batch(codec, batch, code_dir, counters)
            print(f"[{split}] {index + 1}/{len(rows)} encoded={counters['encoded']} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)

        mapped: dict = {"task": row.get("task", "audio_lm"),
                        "source": row.get("source", "unknown"),
                        "lang": row.get("lang", "zh"),
                        "index": index}
        if is_s2s:
            if not pieces.get("answer_audio"):
                counters["failed"] += 1
                continue
            mapped["answer_codes"] = pieces["answer_audio"]
            mapped["prompt_codes"] = pieces.get("prompt_audio")
        else:
            if not pieces.get("audio"):
                counters["failed"] += 1
                continue
            mapped["codes"] = pieces["audio"]
        out_rows.append(mapped)

    flush_batch(codec, batch, code_dir, counters)

    # Drop rows whose referenced shard never got written (encode returned a zero
    # length); otherwise the manifest would point at a missing .npy.
    def shard_ok(rel: str | None) -> bool:
        return bool(rel) and (output / rel).exists()

    kept: list[dict] = []
    for row in out_rows:
        index = row.pop("index")
        if not shard_ok(row.get("answer_codes") or row.get("codes")):
            counters["failed"] += 1
            continue
        if "prompt_codes" in row and row["prompt_codes"] and not shard_ok(row["prompt_codes"]):
            row["prompt_codes"] = None
        kept.append(row)
    out_rows = kept

    manifest_out = output / f"{split}.jsonl"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{split}] wrote {len(out_rows)} rows -> {manifest_out} ({counters})", flush=True)


def main() -> None:
    args = parse_args()
    codec = build_frozen_audio_codec(
        args.codec_type, args.codec_model, args.device, num_codebooks=args.num_codebooks
    )
    print(f"codec: {args.codec_type} Q={codec.num_codebooks} vocab={codec.codebook_size} "
          f"sr={codec.sample_rate} frame_rate={codec.frame_rate_hz:.2f}Hz")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.data.is_dir():
        pairs = [(args.data / "train.jsonl", "train"), (args.data / "dev.jsonl", "dev")]
    else:
        if not args.dev_file:
            raise SystemExit("For a single-file --data, pass --dev-file")
        pairs = [(args.data, "train"), (args.dev_file, "dev")]

    for manifest, split in pairs:
        if not manifest.exists():
            print(f"[{split}] manifest missing, skipped: {manifest}")
            continue
        process_split(args, codec, manifest, split, args.output)

    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "codec_type": args.codec_type,
                "codec_model": args.codec_model,
                "num_codebooks": codec.num_codebooks,
                "codebook_size": codec.codebook_size,
                "sample_rate": codec.sample_rate,
                "frame_rate_hz": codec.frame_rate_hz,
                "audio_offset_note": "LM vocab offset = len(tokenizer); see model/audio_lm.py",
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
