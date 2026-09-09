"""Prepare a randomized VoiceAssistant-400K subset for Speech-MiniMind.

Usage
-----
python scripts/prepare_voiceassistant_400k.py \
  --num-samples 50000 \
  --output data/voiceassistant400k_50k

The script does three jobs:

1) download / load the HF dataset rows,
2) randomly sample a fixed count,
3) export `train.jsonl` and `dev.jsonl` plus downloaded local WAV audio.

Each JSONL item uses the project convention:

- `audio`: local path to wav (relative to output directory)
- `instruction`: text prompt for this sample
- `answer`: target transcript text
- `task`: `voiceassistant_400k`
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tqdm import tqdm

try:
    from datasets import Audio, load_dataset
except ModuleNotFoundError as error:
    raise SystemExit("请先安装 datasets: python -m pip install datasets") from error

try:
    import numpy as np
    import soundfile as sf
except ModuleNotFoundError as error:
    raise SystemExit("请先安装 soundfile: python -m pip install soundfile") from error


QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
TASK_NAME = "voiceassistant_400k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gpt-omni/VoiceAssistant-400K", help="Hugging Face dataset id")
    parser.add_argument("--split", default="train", help="HF split name")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/voiceassistant400k_50k"),
        help="output directory with train.jsonl / dev.jsonl / audio files",
    )
    parser.add_argument("--num-samples", type=int, default=50000, help="number of examples to sample randomly")
    parser.add_argument("--dev-ratio", type=float, default=0.05, help="dev split ratio")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--data-dir", default=None, help="alias for --output")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream rows and reservoir-sample instead of loading the whole split",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("random", "head"),
        default="random",
        help="random: uniform reservoir sample (requires full pass); head: take first N quickly (validation only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent audio download threads (raise for more parallelism)",
    )
    parser.add_argument(
        "--source",
        choices=("hf", "modelscope"),
        default="hf",
        help="hf: HuggingFace hub (needs reachable HF); modelscope: ModelScope parquet mirror (reachable in CN)",
    )
    parser.add_argument(
        "--modelscope-id",
        default="OmniData/VoiceAssistant-400K",
        help="ModelScope dataset id when --source=modelscope",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=1447,
        help="rows per parquet shard for ModelScope source (used to pick how many shards to download)",
    )
    return parser.parse_args()


def _extract_audio_source(audio_field: Any) -> Any:
    if isinstance(audio_field, list):
        for item in audio_field:
            if item:
                return item
        return None
    return audio_field


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_remote_url(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _write_wav(audio: np.ndarray, sample_rate: int, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Explicit format so temporary .part targets still decode correctly.
    sf.write(target, data, int(sample_rate), subtype="PCM_16", format="WAV")


def _copy_or_convert_audio(source: Path, target: Path) -> Path | None:
    source = Path(source)
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav":
        shutil.copy2(source, target)
        return target

    # convert non-wav local audio into wav so existing pipeline can read it
    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
        _write_wav(audio, int(sample_rate), target)
        return target
    except Exception:
        return None


def _download_audio(source_url: str, target: Path) -> Path | None:
    # local cache: a previously converted wav is reused on re-runs
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        req = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as response:
            payload = response.read()
        with BytesIO(payload) as handle:
            audio, sample_rate = sf.read(handle, dtype="float32", always_2d=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(audio, int(sample_rate), tmp)
        tmp.replace(target)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def _write_audio_bytes(raw: bytes, target: Path) -> Path | None:
    """Write encoded audio bytes (e.g. wav bytes from parquet) to `target` as 16-bit PCM wav."""
    if not raw:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with BytesIO(raw) as handle:
            audio, sample_rate = sf.read(handle, dtype="float32", always_2d=False)
        _write_wav(audio, int(sample_rate), tmp)
        tmp.replace(target)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def _resolve_audio_path(audio_field: Any, target: Path) -> Path | None:
    audio_field = _extract_audio_source(audio_field)
    if audio_field is None:
        return None

    # HF question_audio may be dict (path / src / array / sampling_rate) or local path.
    if isinstance(audio_field, dict):
        if raw_bytes := audio_field.get("bytes"):
            # Audio bytes embedded directly (e.g. parquet audio column). Prefer this
            # over path since it requires no extra download.
            if _write_audio_bytes(raw_bytes, target):
                return target

        if raw_path := audio_field.get("path"):
            local = Path(str(raw_path))
            if local.exists():
                return _copy_or_convert_audio(local, target)
            if _is_remote_url(raw_path):
                return _download_audio(raw_path, target)

        if array := audio_field.get("array"):
            sample_rate = audio_field.get("sampling_rate", 16_000)
            try:
                _write_wav(array, int(sample_rate), target)
                return target
            except Exception:
                return None

        if src := audio_field.get("src"):
            if _is_remote_url(src):
                return _download_audio(src, target)
            local = Path(str(src))
            if local.exists():
                return _copy_or_convert_audio(local, target)

    if isinstance(audio_field, str):
        path = Path(audio_field)
        if path.exists():
            return _copy_or_convert_audio(path, target)

    return None


def _write_split(records, split: str, output_dir: Path, audio_dir: Path, start_index: int, workers: int) -> tuple[int, int]:
    output_file = output_dir / f"{split}.jsonl"
    saved = 0
    skipped = 0

    # Each pending item keeps its original row so audio resolution runs in worker threads.
    pending: list[tuple[int, Any, str, str, Path]] = []
    for local_index, row in enumerate(records):
        instruction = _safe_text(row.get(QUESTION_FIELD))
        answer = _safe_text(row.get(ANSWER_FIELD))
        if not instruction or not answer:
            skipped += 1
            continue
        audio_target = audio_dir / f"{split}_{start_index + local_index:06d}.wav"
        pending.append((local_index, row, instruction, answer, audio_target))

    def _resolve(pending_item: tuple[int, Any, str, str, Path]) -> tuple[int, str, str, Path | None]:
        local_index, row, instruction, answer, target = pending_item
        audio_path = _resolve_audio_path(row.get("question_audio"), target)
        return local_index, instruction, answer, audio_path

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(_resolve, pending), total=len(pending), desc=f"audio {split}", unit="item"))
    else:
        results = [tqdm(_resolve(item), desc=f"audio {split}", unit="item") for item in pending]

    # Sort back into original order so jsonl lines are stable.
    results.sort(key=lambda item: item[0])

    with output_file.open("w", encoding="utf-8") as handle, tqdm(total=len(results), desc=f"write {split}", unit="line") as bar:
        for _, instruction, answer, audio_path in results:
            bar.update(1)
            if audio_path is None:
                skipped += 1
                continue
            item = {
                "audio": str(audio_path.relative_to(output_dir)),
                "instruction": instruction,
                "answer": answer,
                "task": TASK_NAME,
                "source": "gpt-omni/VoiceAssistant-400K",
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            saved += 1

    return saved, skipped


def _sample_dataset(dataset, requested: int, seed: int, stream: bool, sample_mode: str) -> Any:
    """Return `requested` rows sampled from the dataset."""
    rng = random.Random(seed)
    if sample_mode == "head":
        # Fast validation path: first N rows, no full pass required.
        rows: list[Any] = []
        for row in dataset:
            rows.append(row)
            if len(rows) >= requested:
                break
        if not rows:
            raise SystemExit("数据集为空或请求数为 0，无法导出")
        return rows, None
    if stream:
        # Reservoir sampling: one pass, O(requested) memory, uniform random.
        reservoir: list[Any] = []
        seen = 0
        for row in dataset:
            seen += 1
            if len(reservoir) < requested:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < requested:
                    reservoir[j] = row
        if seen == 0:
            raise SystemExit("数据集为空或请求数为 0，无法导出")
        return reservoir, seen
    available = len(dataset)
    requested = min(requested, available)
    if requested == 0:
        raise SystemExit("数据集为空或请求数为 0，无法导出")
    sampled = dataset.shuffle(seed=seed).select(range(requested))
    return [sampled[index] for index in range(requested)], available


def _modelscope_file_url(dataset_id: str, file_path: str) -> str:
    return (
        f"https://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo"
        f"?Revision=master&FilePath={file_path}"
    )


def _modelscope_download(url: str, target: Path) -> Path | None:
    """Download `url` to `target`, resuming/reusing cached copies."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=300) as response:
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp.replace(target)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def _modelscope_sample_rows(
    dataset_id: str,
    num_shards: int,
    total_shards: int,
    rows_per_shard: int,
    seed: int,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Randomly pick shard indexes, download those parquets, and return their rows.

    Shards are sampled without replacement and rows from all selected shards are
    concatenated. Each returned row is a plain dict with standard keys so the
    shared export helpers can consume them.
    """
    rng = random.Random(seed)
    shard_indexes = sorted(rng.sample(range(total_shards), k=num_shards))
    rows: list[dict[str, Any]] = []
    for shard in tqdm(shard_indexes, desc="download parquet", unit="shard"):
        fname = f"train-{shard:05d}-of-{total_shards:05d}.parquet"
        url = _modelscope_file_url(dataset_id, f"data/{fname}")
        local = _modelscope_download(url, cache_dir / fname)
        if local is None:
            print(f"WARN: failed to download {fname}, skipping")
            continue
        import pyarrow.parquet as pq  # local import keeps hf-path startup light

        table = pq.read_table(local)
        for batch in table.to_batches():
            for record in batch.to_pylist():
                rows.append(record)
    return rows


def _copy_rows_for_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize rows with the fields the exporter expects (in place)."""
    return rows


def _modelscope_main(args: argparse.Namespace, output_dir: Path, audio_dir: Path) -> None:
    import json as _json

    cache_dir = output_dir / "_parquet_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine shard count / layout from the ModelScope tree API.
    probe_url = f"https://www.modelscope.cn/api/v1/datasets/{args.modelscope_id}/repo/tree?Revision=master&Root=data"
    req = Request(probe_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        payload = _json.loads(response.read().decode("utf-8"))
    total_shards = payload["Data"]["TotalCount"]
    if total_shards is None or total_shards <= 0:
        raise SystemExit("无法从 ModelScope 读取分片数量")

    rows_per_shard = args.shard_rows

    num_available = total_shards * rows_per_shard
    requested = max(0, min(args.num_samples, num_available))
    if requested == 0:
        raise SystemExit("数据集为空或请求数为 0，无法导出")

    # Enough shards to cover requested rows (rounding up), but not more than available.
    needed_shards = max(1, min(int((requested + rows_per_shard - 1) // rows_per_shard), total_shards))

    rows = _modelscope_sample_rows(
        args.modelscope_id,
        needed_shards,
        total_shards,
        rows_per_shard,
        args.seed,
        cache_dir,
    )
    # Trim to exact requested count in stable order.
    rows = rows[:requested]

    dev_count = max(0, min(len(rows), int(len(rows) * args.dev_ratio)))
    train_count = max(1, len(rows) - dev_count) if len(rows) > 1 else 1
    if train_count + dev_count > len(rows):
        dev_count = len(rows) - train_count

    train_dataset = rows[:train_count]
    dev_dataset = rows[train_count : train_count + dev_count]

    train_saved, train_skipped = _write_split(train_dataset, "train", output_dir, audio_dir, 0, args.workers)
    dev_saved, dev_skipped = _write_split(dev_dataset, "dev", output_dir, audio_dir, train_count, args.workers)

    summary = {
        "dataset": args.modelscope_id,
        "source": "modelscope",
        "split": args.split,
        "requested": requested,
        "shards_downloaded": needed_shards,
        "shard_rows": rows_per_shard,
        "seed": args.seed,
        "train_samples": train_saved,
        "dev_samples": dev_saved,
        "train_skipped": train_skipped,
        "dev_skipped": dev_skipped,
        "dev_ratio": args.dev_ratio,
        "total_saved": train_saved + dev_saved,
        "output": str(output_dir),
        "note": "audio bytes embedded in parquet, converted to 16-bit PCM wav; parquet cached under _parquet_cache",
    }

    (output_dir / "metadata.json").write_text(_json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train_jsonl: {output_dir / 'train.jsonl'} ({train_saved} samples)")
    print(f"dev_jsonl: {output_dir / 'dev.jsonl'} ({dev_saved} samples)")
    print(f"metadata: {output_dir / 'metadata.json'}")
    print(_json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.data_dir is not None:
        args.output = Path(args.data_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "modelscope":
        _modelscope_main(args, output_dir, audio_dir)
        return

    # ---- HF path ----
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.stream)
    # Keep question_audio as an undecoded reference in both modes. Decoding on the
    # fly would require extra decoders (e.g. torchcodec) and is unnecessary here.
    dataset = dataset.cast_column("question_audio", Audio(decode=False))

    sampled, available = _sample_dataset(dataset, args.num_samples, args.seed, args.stream, args.sample_mode)

    dev_count = max(0, min(len(sampled), int(len(sampled) * args.dev_ratio)))
    train_count = max(1, len(sampled) - dev_count) if len(sampled) > 1 else 1
    if train_count + dev_count > len(sampled):
        dev_count = len(sampled) - train_count

    train_dataset = sampled[:train_count]
    dev_dataset = sampled[train_count : train_count + dev_count]

    train_saved, train_skipped = _write_split(train_dataset, "train", output_dir, audio_dir, 0, args.workers)
    dev_saved, dev_skipped = _write_split(dev_dataset, "dev", output_dir, audio_dir, train_count, args.workers)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "streaming": args.stream,
        "available": available,
        "requested": len(sampled),
        "seed": args.seed,
        "workers": args.workers,
        "train_samples": train_saved,
        "dev_samples": dev_saved,
        "train_skipped": train_skipped,
        "dev_skipped": dev_skipped,
        "dev_ratio": args.dev_ratio,
        "total_saved": train_saved + dev_saved,
        "output": str(output_dir),
        "note": "question_audio is resolved concurrently (workers) and cached to wav; re-runs skip already-downloaded files",
    }

    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train_jsonl: {output_dir / 'train.jsonl'} ({train_saved} samples)")
    print(f"dev_jsonl: {output_dir / 'dev.jsonl'} ({dev_saved} samples)")
    print(f"metadata: {output_dir / 'metadata.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
