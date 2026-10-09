"""Prepare SenseVoice waveform inputs and existing assistant Mimi codes from sft_a2a.

No question codec encoding or model downloads are needed. --download optionally
fetches only the explicitly requested dataset. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from dataset.thinker_talker_s2s import load_waveform  # noqa: E402

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
    parser.add_argument("--lang", default="zh", choices=("zh", "en", "all"),
                        help="keep rows whose assistant reply is mostly this language")
    parser.add_argument("--dev-rows", type=int, default=200,
                        help="rows held out for dev, sampled from the tail")
    parser.add_argument("--max-answer-frames", type=int, default=750,
                        help="drop rows whose answer exceeds this many frames")
    parser.add_argument("--limit", type=int, default=0, help="only process N rows (smoke test)")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> Path:
    """Fetch the parquet from ModelScope.

    ``gongjy/minimind-o_dataset`` is a *dataset* repo, so the default
    ``snapshot_download(..., repo_type='model')`` 404s; ask for the dataset
    endpoint explicitly (with a fallback for older ``modelscope`` versions that
    lack ``dataset_snapshot_download``).
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        try:
            from modelscope.hub.snapshot_download import dataset_snapshot_download
        except ImportError:
            dataset_snapshot_download = None
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "modelscope is required for --download. Run: python -m pip install modelscope"
        ) from error

    if dataset_snapshot_download is not None:
        local_dir = dataset_snapshot_download(
            args.dataset_id, local_dir=str(args.download_dir),
            allow_patterns=[args.file_name],
        )
    else:
        local_dir = snapshot_download(
            args.dataset_id, local_dir=str(args.download_dir),
            repo_type="dataset", allow_patterns=[args.file_name],
        )
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
    if waveform.size == 0 or not np.isfinite(waveform).all():
        return None
    return waveform.astype(np.float32), int(rate)


def main() -> None:
    import soundfile as sf
    args = parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; choose a new directory")
    if args.dev_rows < 1 or args.max_answer_frames < 1 or args.limit < 0:
        raise SystemExit("dev-rows/max-answer-frames must be positive; limit nonnegative")
    parquet_path = args.parquet
    if args.download:
        parquet_path = download_dataset(args)
    if parquet_path is None or not parquet_path.is_file():
        raise SystemExit("provide an existing --parquet or explicitly request --download")
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(str(parquet_path))
    required = ["conversations", "answer_audios", "question_audios"]
    if not set(required).issubset(parquet.schema_arrow.names):
        raise SystemExit("parquet requires conversations, answer_audios and question_audios")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "codes").mkdir()
    (output / "audio").mkdir()
    rows = []
    stats = {"rows": 0, "kept": 0, "rejected": 0}
    for batch in parquet.iter_batches(batch_size=64, columns=required):
        for record in batch.to_pylist():
            if args.limit and stats["rows"] >= args.limit:
                break
            index = stats["rows"]
            stats["rows"] += 1
            conversations = record["conversations"]
            if isinstance(conversations, str):
                conversations = json.loads(conversations)
            users = [t for t in conversations if t.get("role") == "user"]
            assistants = [t for t in conversations if t.get("role") == "assistant"]
            questions = record["question_audios"] or []
            answers = record["answer_audios"] or []
            prompt = str(users[-1].get("content", "")).strip() if users else ""
            answer = str(assistants[-1].get("content", "")).strip() if assistants else ""
            codes = split_answer_tokens(answers[-1], 8, 2048) if answers else None
            decoded = decode_audio_bytes(questions[-1]) if questions and questions[-1] else None
            if (not prompt or not answer or not keep_language(answer, args.lang)
                    or codes is None or codes.shape[1] > args.max_answer_frames or decoded is None):
                stats["rejected"] += 1
                continue
            audio_rel = f"audio/{index:07d}.wav"
            # Preserve the decoded mono question at its original rate; the shared
            # loader resamples each utterance independently to 16 kHz at use time.
            sf.write(output / audio_rel, decoded[0], decoded[1], subtype="FLOAT")
            load_waveform(output / audio_rel)  # validate the exact artifact the trainer reads
            code_rel = f"codes/{index:07d}_a.npy"
            np.save(output / code_rel, codes)
            rows.append(dict(prompt_audio=audio_rel, prompt_text=prompt, answer_text=answer,
                             answer_codes=code_rel, frames=int(codes.shape[1]), task="speech_qa"))
            stats["kept"] += 1
        if args.limit and stats["rows"] >= args.limit:
            break
    if len(rows) < 2:
        raise SystemExit("need at least two valid pairs for nonempty train/dev; inspect output artifacts")
    split_at = max(1, len(rows) - args.dev_rows)
    for split, subset in (("train", rows[:split_at]), ("dev", rows[split_at:])):
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset), encoding="utf-8")
        print(f"{split}: {len(subset)} rows")
    metadata = dict(source_parquet=str(parquet_path), codec_type="mimi", num_codebooks=8,
                    codebook_size=2048, input_type="sensevoice_waveform", stats=stats)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
