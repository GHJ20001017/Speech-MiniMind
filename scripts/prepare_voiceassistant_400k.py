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
import shutil
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
    sf.write(target, data, int(sample_rate), subtype="PCM_16")


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
    try:
        req = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as response:
            payload = response.read()
        with BytesIO(payload) as handle:
            audio, sample_rate = sf.read(handle, dtype="float32", always_2d=False)
        _write_wav(audio, int(sample_rate), target)
        return target
    except Exception:
        return None


def _resolve_audio_path(audio_field: Any, target: Path) -> Path | None:
    audio_field = _extract_audio_source(audio_field)
    if audio_field is None:
        return None

    # HF question_audio may be dict (path / src / array / sampling_rate) or local path.
    if isinstance(audio_field, dict):
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


def _write_split(records, split: str, output_dir: Path, audio_dir: Path, start_index: int) -> tuple[int, int]:
    output_file = output_dir / f"{split}.jsonl"
    saved = 0
    skipped = 0

    with output_file.open("w", encoding="utf-8") as handle, tqdm(total=len(records), desc=f"write {split}") as bar:
        for local_index, row in enumerate(records):
            bar.update(1)
            instruction = _safe_text(row.get(QUESTION_FIELD))
            answer = _safe_text(row.get(ANSWER_FIELD))
            if not instruction or not answer:
                skipped += 1
                continue

            audio_target = audio_dir / f"{split}_{start_index + local_index:06d}.wav"
            audio_path = _resolve_audio_path(row.get("question_audio"), audio_target)
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

            for key in ("split_name", "index", "round"):
                value = _safe_text(row.get(key))
                if value:
                    item[key] = value

            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            saved += 1

    return saved, skipped


def main() -> None:
    args = parse_args()
    if args.data_dir is not None:
        args.output = Path(args.data_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.cast_column("question_audio", Audio(decode=False))

    available = len(dataset)
    requested = max(0, min(args.num_samples, available))
    if requested == 0:
        raise SystemExit("数据集为空或请求数为 0，无法导出")

    sampled = dataset.shuffle(seed=args.seed).select(range(requested))

    dev_count = max(0, min(requested, int(requested * args.dev_ratio)))
    train_count = max(1, requested - dev_count) if requested > 1 else 1
    if train_count + dev_count > requested:
        dev_count = requested - train_count

    train_dataset = sampled.select(range(train_count))
    dev_dataset = sampled.select(range(train_count, train_count + dev_count))

    train_saved, train_skipped = _write_split(train_dataset, "train", output_dir, audio_dir, 0)
    dev_saved, dev_skipped = _write_split(dev_dataset, "dev", output_dir, audio_dir, train_count)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "requested": requested,
        "seed": args.seed,
        "train_samples": train_saved,
        "dev_samples": dev_saved,
        "train_skipped": train_skipped,
        "dev_skipped": dev_skipped,
        "dev_ratio": args.dev_ratio,
        "total_saved": train_saved + dev_saved,
        "output": str(output_dir),
        "note": "question_audio is handled by download/copy and converted to wav when needed",
    }

    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train_jsonl: {output_dir / 'train.jsonl'} ({train_saved} samples)")
    print(f"dev_jsonl: {output_dir / 'dev.jsonl'} ({dev_saved} samples)")
    print(f"metadata: {output_dir / 'metadata.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
