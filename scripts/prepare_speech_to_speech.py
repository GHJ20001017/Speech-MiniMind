"""Build a Route-B speech-to-speech manifest from a MiniMind-O ``sft_a2a`` parquet.

MiniMind-O publishes its audio-to-audio SFT set as ``sft_a2a.parquet`` on
ModelScope (``gongjy/minimind-o_dataset``).  Each row already carries the
tokenised answer audio, so we only need to:

1. read the parquet (``conversations`` / ``question_audios`` / ``answer_audios``),
2. turn the flattened answer tokens back into a ``(8, T)`` Mimi code array,
3. encode the *question* audio bytes with the same frozen Mimi codec, and
4. write both as ``.npy`` shards plus a JSONL manifest the trainer reads.

Tokens in ``answer_audios`` are stored flat as 8 codes per frame, with a
per-layer stop token (``>= codebook_size``) terminating each layer; those are
truncated here because the trainer appends ``<|audio_eos|>`` itself.

Usage::

    # download once (needs `modelscope`), then convert
    python scripts/prepare_speech_to_speech.py --download --output data/route_b/s2s
    python scripts/prepare_speech_to_speech.py \
        --parquet /path/to/sft_a2a.parquet --output data/route_b/s2s --lang zh

The script is CPU-capable but the question-audio encoding is much faster on a
GPU; pass ``--device cuda:0`` and ``--batch-size 8`` on the training host.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from model.audio_codec import build_frozen_audio_codec  # noqa: E402

DATASET_ID = "gongjy/minimind-o_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=None,
                        help="path to sft_a2a.parquet (or sft_a2a_mini.parquet)")
    parser.add_argument("--download", action="store_true",
                        help="download the dataset from ModelScope first")
    parser.add_argument("--download-dir", type=Path, default=Path("data/route_b/_download"),
                        help="where --download stores the snapshot")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--file-name", default="sft_a2a.parquet",
                        help="parquet file to use inside the dataset repo")
    parser.add_argument("--output", type=Path, default=Path("data/route_b/s2s"))
    parser.add_argument("--codec-type", default="mimi", choices=("mimi", "encodec"))
    parser.add_argument("--codec-model", default=None,
                        help="codec model id or local dir (defaults per backend)")
    parser.add_argument("--device", default="cpu", help="torch device for codec encoding")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lang", default="zh", choices=("zh", "en", "all"),
                        help="keep rows whose assistant reply is mostly this language")
    parser.add_argument("--dev-rows", type=int, default=200,
                        help="rows held out for dev, sampled from the tail")
    parser.add_argument("--max-answer-frames", type=int, default=750,
                        help="drop rows whose answer exceeds this many frames")
    parser.add_argument("--limit", type=int, default=0, help="only process N rows (smoke test)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-prompt-encode", action="store_true",
                        help="do not encode question audio (writes prompt_codes=null)")
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "modelscope is required for --download. Run: python -m pip install modelscope"
        ) from error
    local_dir = snapshot_download(args.dataset_id, local_dir=str(args.download_dir))
    candidate = Path(local_dir) / args.file_name
    if not candidate.exists():
        available = sorted(p.name for p in Path(local_dir).glob("*.parquet"))
        raise SystemExit(
            f"{candidate} not found. Parquet files in the snapshot: {available}"
        )
    return candidate


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return cjk / len(text)


def keep_language(assistant_text: str, lang: str) -> bool:
    if lang == "all":
        return True
    ratio = cjk_ratio(assistant_text)
    return ratio >= 0.15 if lang == "zh" else ratio < 0.15


def split_answer_tokens(tokens: list[int], num_codebooks: int, codebook_size: int) -> np.ndarray | None:
    """``[t0q0..t0q7, t1q0.., ...]`` -> ``(Q, T)`` int16, stopping at specials."""
    usable: list[list[int]] = [[] for _ in range(num_codebooks)]
    for frame_start in range(0, len(tokens) - num_codebooks + 1, num_codebooks):
        frame = tokens[frame_start : frame_start + num_codebooks]
        if any(not (0 <= int(code) < codebook_size) for code in frame):
            break  # stop token / corruption: end of the answer
        for codebook in range(num_codebooks):
            usable[codebook].append(int(frame[codebook]))
    length = len(usable[0])
    if length == 0 or any(len(layer) != length for layer in usable):
        return None
    return np.asarray(usable, dtype=np.int16)


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int] | None:
    import soundfile as sf

    try:
        waveform, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - per-row tolerance
        return None
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        return None
    return waveform.astype(np.float32), int(rate)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    parquet_path = args.parquet
    if args.download or parquet_path is None:
        parquet_path = download_dataset(args)
    if not parquet_path.exists():
        raise SystemExit(f"parquet not found: {parquet_path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pyarrow is required to read the parquet. Run: python -m pip install pyarrow"
        ) from error

    codec = None
    if not args.skip_prompt_encode:
        codec = build_frozen_audio_codec(args.codec_type, args.codec_model, args.device)

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    for required in ("conversations", "answer_audios"):
        if required not in columns:
            raise SystemExit(f"parquet is missing required column '{required}' (has {sorted(columns)})")
    has_questions = "question_audios" in columns

    num_codebooks = codec.num_codebooks if codec is not None else 8
    codebook_size = codec.codebook_size if codec is not None else 2048

    output: Path = args.output
    code_dir = output / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stats = {"rows": 0, "kept": 0, "no_answer": 0, "bad_answer": 0, "lang": 0,
             "too_long": 0, "no_question": 0, "question_failed": 0}

    # Collect candidates first so the codec can encode question audio in batches.
    pending: list[dict] = []
    total = table.num_rows if not args.limit else min(args.limit, table.num_rows)
    for index in range(total):
        stats["rows"] += 1
        conversations = json.loads(table["conversations"][index].as_py())
        assistant_turns = [t for t in conversations if t.get("role") == "assistant"]
        if not assistant_turns:
            stats["no_answer"] += 1
            continue
        if not keep_language(str(assistant_turns[-1].get("content", "")), args.lang):
            stats["lang"] += 1
            continue

        answer_audios = table["answer_audios"][index].as_py() or []
        if not answer_audios:
            stats["no_answer"] += 1
            continue
        codes = split_answer_tokens(
            [int(t) for t in answer_audios[-1]], num_codebooks, codebook_size
        )
        if codes is None:
            stats["bad_answer"] += 1
            continue
        if codes.shape[1] > args.max_answer_frames:
            stats["too_long"] += 1
            continue

        question_audio = None
        if codec is not None and has_questions:
            audios = table["question_audios"][index].as_py() or []
            if audios and audios[-1]:
                question_audio = audios[-1]
            else:
                stats["no_question"] += 1
        pending.append(
            {"index": index, "codes": codes, "audio": question_audio}
        )

    # Batch-encode question audio.
    if codec is not None:
        batch_size = max(1, args.batch_size)
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            loaded = []
            for item in chunk:
                decoded = decode_audio_bytes(item["audio"]) if item["audio"] else None
                loaded.append(decoded)
            valid = [(item, dec) for item, dec in zip(chunk, loaded) if dec is not None]
            if valid:
                waveforms = [torch.from_numpy(dec[0]) for _, dec in valid]
                lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.long)
                padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
                rates = {dec[1] for _, dec in valid}
                rate = rates.pop() if len(rates) == 1 else 16000
                codes, code_lengths = codec.encode(padded, lengths, rate)
                for (item, _), code, length in zip(valid, codes, code_lengths.tolist()):
                    item["question_codes"] = code[:, :length].numpy().astype(np.int16)
            for item in chunk:
                if "question_codes" not in item:
                    stats["question_failed"] += 1
                    item["question_codes"] = None
            print(f"[encode] {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

    split_at = max(1, len(pending) - args.dev_rows)
    for position, item in enumerate(pending):
        split = "train" if position < split_at else "dev"
        np.save(code_dir / f"{item['index']:07d}_a.npy", item["codes"])
        prompt_rel = None
        if item.get("question_codes") is not None:
            np.save(code_dir / f"{item['index']:07d}_p.npy", item["question_codes"])
            prompt_rel = f"codes/{item['index']:07d}_p.npy"
        rows.append(
            {
                "split": split,
                "prompt_codes": prompt_rel,
                "answer_codes": f"codes/{item['index']:07d}_a.npy",
                "task": "speech_qa",
                "source": f"minimind_o/{args.file_name}",
                "lang": args.lang if args.lang != "all" else "mixed",
                "frames": int(item["codes"].shape[1]),
            }
        )
        stats["kept"] += 1

    for split in ("train", "dev"):
        subset = [r for r in rows if r["split"] == split]
        manifest = output / f"{split}.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps({k: v for k, v in row.items() if k != "split"},
                                        ensure_ascii=False) + "\n")
        print(f"{split}: {len(subset)} rows -> {manifest}")

    metadata = {
        "source_parquet": str(parquet_path),
        "codec_type": args.codec_type,
        "codec_model": args.codec_model,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "lang": args.lang,
        "stats": stats,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
