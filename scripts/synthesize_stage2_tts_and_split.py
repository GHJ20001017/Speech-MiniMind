#!/usr/bin/env python3
"""Synthesize missing stage-2 prompt audio and create source-stratified splits.

The stage-2 corpus contains two kinds of rows:

* rows with an existing ``wav`` path (MOSS / VoiceAssistant), kept unchanged;
* text-only rows whose ``wav`` field is empty, requiring TTS.

This script fills the empty ``wav`` fields by synthesizing each row's ``prompt``
with Qwen3-TTS CustomVoice, then writes train/val/test JSONL files.  By default,
every synthesized row is assigned a deterministic random speaker from the nine
speakers bundled with the CustomVoice model.  The chosen speaker is stored in
the row as ``tts_speaker`` and in the WAV path, so an interrupted run resumes
with the same voice for every row.  The split is stratified by ``source``: every
source contributes to every non-empty split.

Design notes
------------
* Resume-friendly: output WAV paths are deterministic, so an interrupted run
  skips already-valid WAV files and retries only missing/corrupt ones.
* Multi-GPU: one worker process per selected GPU, each with its own model copy.
* Batched inference: each worker groups rows by source/speaker/character count
  and sends a list to ``generate_custom_voice``.  A failed batch is split in
  half recursively so one bad row does not discard the whole batch.
* Safe manifest update: the JSONL update is atomic; when updating in place, a
  ``.bak`` file is kept by default.
* No silent partial split: if any TTS item is still missing, the script writes
  the partially updated manifest but exits before producing split files unless
  ``--allow-partial`` is given.

Example on the 95 host
----------------------
    cd /gpu3/guhj/Speech-MiniMind
    /gpu3/guhj/envs/speech-llm/bin/python \
        scripts/synthesize_stage2_tts_and_split.py --gpus 0,1,2,3,4,5

By default this updates ``data/speech2text_corpus/stage2_no_aishell.jsonl`` in
place and writes splits to ``data/speech2text_corpus/splits``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/speech2text_corpus/stage2_no_aishell.jsonl"
DEFAULT_AUDIO_DIR = ROOT / "data/speech2text_corpus/audio"
DEFAULT_SPLITS_DIR = ROOT / "data/speech2text_corpus/splits"
DEFAULT_MODEL = "/gpu3/guhj/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_BATCH_CHAR_RATIO = 1.6
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="stage-2 JSONL manifest")
    parser.add_argument("--output-jsonl", type=Path, default=None,
                        help="updated manifest path; default: update --input in place")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
                        help="directory for synthesized WAV files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Qwen3-TTS CustomVoice model directory")
    parser.add_argument("--gpus", default="auto",
                        help="comma-separated physical GPU ids, or 'auto' to pick idle GPUs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="TTS rows per model call per GPU; use 1 to disable batching")
    parser.add_argument("--max-batch-char-ratio", type=float, default=DEFAULT_MAX_BATCH_CHAR_RATIO,
                        help="maximum longest/shortest prompt character ratio within one batch")
    speaker_group = parser.add_mutually_exclusive_group()
    speaker_group.add_argument("--speaker", default=None,
                               help="use one fixed Qwen3-TTS speaker for all rows")
    speaker_group.add_argument("--speakers", default=None,
                               help="comma-separated random speaker pool; default: all model speakers")
    parser.add_argument("--speaker-seed", type=int, default=42,
                        help="seed for deterministic per-row random speaker assignment")
    parser.add_argument("--language", default="Chinese",
                        help="Qwen3-TTS language label")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N empty-wav rows (smoke test)")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="do not launch TTS workers; only collect existing WAVs and split")
    parser.add_argument("--skip-split", action="store_true",
                        help="update the manifest but do not create train/val/test files")
    parser.add_argument("--allow-partial", action="store_true",
                        help="split non-empty rows even when some TTS items failed")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate WAV files even if valid files already exist")
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True,
                        help="keep a .bak copy when updating the input manifest in place")
    parser.add_argument("--backup-path", type=Path, default=None,
                        help="custom backup path for in-place manifest updates")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="print worker progress every N generated rows")
    parser.add_argument("--dry-run", action="store_true",
                        help="print synthesis/split plan without writing anything")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                        help="output directory for train/val/test JSONL files")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="deterministic shuffle seed for source-stratified splits")
    parser.add_argument("--min-val-per-source", type=int, default=1)
    parser.add_argument("--min-test-per-source", type=int, default=1)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append((line_number, record))
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)


def valid_wav(path: Path) -> bool:
    """Return True for a readable, non-empty PCM WAV file."""
    try:
        if not path.is_file() or path.stat().st_size <= 44:
            return False
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() > 0 and handle.getframerate() > 0
    except (EOFError, OSError, wave.Error):
        return False


def stable_audio_key(source: str, prompt: str) -> str:
    digest = hashlib.sha1()
    digest.update(source.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()[:20]


def stable_speaker(source: str, prompt: str, speakers: tuple[str, ...], seed: int) -> str:
    """Choose a stable pseudo-random speaker for one source/prompt row."""
    if not speakers:
        raise ValueError("speaker pool must not be empty")
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(source.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    return speakers[int(digest.hexdigest()[:16], 16) % len(speakers)]


def safe_source_name(source: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", source.strip())
    return value.strip("._-") or "unknown"


def manifest_audio_path(audio_path: Path) -> str:
    """Use repo-relative paths so the same manifest works from the repo root."""
    resolved = audio_path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_targets(
    rows: list[tuple[int, dict[str, Any]]],
    audio_dir: Path,
    limit: int,
    speakers: tuple[str, ...],
    speaker_seed: int,
) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for row_index, (_, record) in enumerate(rows):
        if str(record.get("wav") or "").strip():
            continue
        prompt = str(record.get("prompt") or "").strip()
        source = str(record.get("source") or "unknown").strip() or "unknown"
        if not prompt:
            print(f"[warning] row {row_index + 1} has empty prompt; leaving wav empty", file=sys.stderr)
            continue
        speaker = stable_speaker(source, prompt, speakers, speaker_seed)
        audio_path = (
            audio_dir
            / safe_source_name(source)
            / safe_source_name(speaker)
            / f"{stable_audio_key(source, prompt)}.wav"
        )
        targets.append(
            {
                "row_index": str(row_index),
                "source": source,
                "speaker": speaker,
                "prompt": prompt,
                "path": str(audio_path),
                "manifest_path": manifest_audio_path(audio_path),
            }
        )
        if limit > 0 and len(targets) >= limit:
            break
    return targets


def resolve_gpus(spec: str) -> list[int]:
    spec = spec.strip()
    if spec.lower() != "auto":
        gpus = [int(item) for item in spec.split(",") if item.strip()]
        if not gpus:
            raise ValueError("--gpus must contain at least one GPU id")
        return gpus

    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value and env_value != "-1":
        gpus = [int(item) for item in env_value.split(",") if item.strip()]
        if gpus:
            return gpus

    query = "index,memory.used,memory.total,utilization.gpu"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot auto-detect GPUs; pass --gpus explicitly") from exc

    candidates: list[tuple[int, int]] = []
    all_gpus: list[tuple[int, int, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, used_mb, total_mb, util = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        all_gpus.append((index, used_mb, total_mb))
        if used_mb <= 10_000 and util <= 20:
            candidates.append((index, used_mb))
    if candidates:
        return [gpu for gpu, _ in sorted(candidates)]
    if all_gpus:
        gpu, _ = min(all_gpus, key=lambda item: item[1])
        print(f"[warning] no clearly idle GPU found; using the least-used GPU {gpu}", file=sys.stderr)
        return [gpu]
    raise RuntimeError("nvidia-smi returned no GPUs")


def write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    array = np.asarray(audio)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise RuntimeError(f"unexpected waveform shape: {array.shape}")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    sf.write(str(tmp_path), array, sample_rate, format="WAV", subtype="PCM_16")
    if not valid_wav(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("written WAV is empty or unreadable")
    os.replace(tmp_path, path)


def log_synthesis_error(error_path: Path, item: dict[str, str], gpu: int, error: BaseException) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "gpu": gpu,
        "row_index": int(item["row_index"]),
        "source": item["source"],
        "speaker": item["speaker"],
        "path": item["path"],
        "prompt": item["prompt"][:500],
        "error": repr(error),
    }
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def synthesize_batch(
    model: Any,
    batch: list[dict[str, str]],
    language: str,
    gpu: int,
    overwrite: bool,
    error_path: Path,
) -> tuple[int, int, int]:
    """Synthesize one batch, recursively splitting it when the model call fails."""
    pending = [
        item for item in batch
        if overwrite or not valid_wav(Path(item["path"]))
    ]
    skipped = len(batch) - len(pending)
    if not pending:
        return 0, skipped, 0

    try:
        wavs, sample_rate = model.generate_custom_voice(
            text=[item["prompt"] for item in pending],
            speaker=[item["speaker"] for item in pending],
            language=[language] * len(pending),
        )
        if len(wavs) != len(pending):
            raise RuntimeError(f"batch output count mismatch: {len(wavs)} != {len(pending)}")
        done = 0
        for item, audio in zip(pending, wavs):
            write_wav(Path(item["path"]), audio, sample_rate)
            done += 1
        return done, skipped, 0
    except Exception as exc:  # noqa: BLE001
        if len(pending) > 1:
            log_synthesis_error(error_path, pending[0], gpu, RuntimeError(f"batch of {len(pending)} failed: {exc!r}"))
            midpoint = len(pending) // 2
            left_done, left_skipped, left_failed = synthesize_batch(
                model, pending[:midpoint], language, gpu, overwrite, error_path
            )
            right_done, right_skipped, right_failed = synthesize_batch(
                model, pending[midpoint:], language, gpu, overwrite, error_path
            )
            return (
                left_done + right_done,
                skipped + left_skipped + right_skipped,
                left_failed + right_failed,
            )
        log_synthesis_error(error_path, pending[0], gpu, exc)
        return 0, skipped, 1


def make_batches(targets: list[dict[str, str]], batch_size: int, max_char_ratio: float) -> list[list[dict[str, str]]]:
    """Build length-aware batches without mixing very long and very short prompts."""
    ordered = sorted(targets, key=lambda item: (item["source"], item["speaker"], len(item["prompt"])))
    batches: list[list[dict[str, str]]] = []
    batch: list[dict[str, str]] = []
    shortest = 0
    for item in ordered:
        length = len(item["prompt"])
        too_long = bool(batch) and max(shortest, length) / max(1, min(shortest, length)) > max_char_ratio
        if batch and (len(batch) >= batch_size or too_long):
            batches.append(batch)
            batch = []
        if not batch:
            shortest = length
        batch.append(item)
    if batch:
        batches.append(batch)
    return batches


def worker(gpu: int, targets: list[dict[str, str]], model_path: str,
           language: str, batch_size: int, max_char_ratio: float, overwrite: bool,
           progress_every: int, error_log: str) -> None:
    """Synthesize one GPU's assigned rows."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn is not installed on the host
    )

    done = 0
    skipped = 0
    failed = 0
    started = time.time()
    error_path = Path(error_log)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    # Similar text lengths produce similar audio lengths; grouping keeps padded
    # batch compute low and also improves tokenizer/codec locality.
    for batch in make_batches(targets, batch_size, max_char_ratio):
        batch_done, batch_skipped, batch_failed = synthesize_batch(
            model, batch, language, gpu, overwrite, error_path
        )
        done += batch_done
        skipped += batch_skipped
        failed += batch_failed
        if done and progress_every > 0 and done % progress_every < batch_done:
            elapsed = time.time() - started
            print(
                f"[gpu {gpu}] done={done} skipped={skipped} failed={failed} "
                f"rate={done / max(elapsed, 1e-9):.2f}/s",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"[gpu {gpu}] finished done={done} skipped={skipped} failed={failed} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def run_synthesis(args: argparse.Namespace, targets: list[dict[str, str]], gpus: list[int]) -> tuple[list[int], Path]:
    context = mp.get_context("spawn")
    log_dir = args.audio_dir / ".tts_logs" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[list[dict[str, str]]] = [[] for _ in gpus]
    for index, item in enumerate(targets):
        chunks[index % len(gpus)].append(item)

    processes: list[mp.Process] = []
    for gpu, chunk in zip(gpus, chunks):
        process = context.Process(
            target=worker,
            args=(
                gpu,
                chunk,
                args.model,
                args.language,
                args.batch_size,
                args.max_batch_char_ratio,
                args.overwrite,
                args.progress_every,
                str(log_dir / f"errors_gpu_{gpu}.jsonl"),
            ),
        )
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        raise

    return [int(process.exitcode or 0) for process in processes], log_dir


def read_failure_count(log_dir: Path) -> int:
    count = 0
    for path in log_dir.glob("errors_gpu_*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def split_counts(
    total: int,
    ratios: dict[str, float],
    min_counts: dict[str, int],
) -> dict[str, int]:
    """Allocate `total` rows proportionally, preserving requested minimums."""
    if total <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    counts = {name: math.floor(total * ratios[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    fractions = sorted(
        SPLIT_NAMES,
        key=lambda name: (total * ratios[name] - counts[name], name),
        reverse=True,
    )
    for name in fractions[:remainder]:
        counts[name] += 1

    for name in SPLIT_NAMES:
        target_min = min(min_counts.get(name, 0), total)
        while counts[name] < target_min:
            donors = [
                donor for donor in SPLIT_NAMES
                if counts[donor] > max(0, min_counts.get(donor, 0))
                and counts[donor] > counts[name]
            ]
            if not donors:
                break
            donor = max(donors, key=lambda item: counts[item])
            counts[donor] -= 1
            counts[name] += 1

    if sum(counts.values()) != total:
        raise RuntimeError(f"internal split allocation error: {counts} for {total}")
    return counts


def source_stratified_split(
    rows: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
    min_counts: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if str(record.get("wav") or "").strip():
            grouped[str(record.get("source") or "unknown")].append(record)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    by_source: dict[str, dict[str, int]] = {}
    for source in sorted(grouped):
        source_rows = grouped[source]
        rng.shuffle(source_rows)
        counts = split_counts(len(source_rows), ratios, min_counts)
        by_source[source] = counts
        offset = 0
        for name in SPLIT_NAMES:
            count = counts[name]
            splits[name].extend(source_rows[offset:offset + count])
            offset += count

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    metadata = {
        "ratios": ratios,
        "seed": seed,
        "counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "counts_by_source": by_source,
    }
    return splits, metadata


def main() -> int:
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise SystemExit("split ratios must be non-negative")
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(f"split ratios must sum to 1.0, got {ratio_sum}")
    if args.train_ratio <= 0:
        raise SystemExit("--train-ratio must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_batch_char_ratio < 1.0:
        raise SystemExit("--max-batch-char-ratio must be >= 1.0")

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"input manifest does not exist: {input_path}")
    output_jsonl = (args.output_jsonl or args.input).resolve()
    args.audio_dir = args.audio_dir.resolve()
    args.splits_dir = args.splits_dir.resolve()

    loaded_rows = load_jsonl(input_path)
    records = [record for _, record in loaded_rows]
    configured_speakers = (
        (args.speaker,)
        if args.speaker
        else tuple(item.strip() for item in (args.speakers or ",".join(DEFAULT_SPEAKERS)).split(",") if item.strip())
    )
    if not configured_speakers:
        raise SystemExit("speaker pool must not be empty")
    targets = build_targets(
        loaded_rows,
        args.audio_dir,
        args.limit,
        configured_speakers,
        args.speaker_seed,
    )

    source_target_counts = Counter(item["source"] for item in targets)
    if args.dry_run:
        print(f"input: {input_path}")
        print(f"updated JSONL: {output_jsonl}")
        print(f"audio dir: {args.audio_dir}")
        print(f"rows: {len(records)}")
        print(f"empty-wav rows selected for TTS: {len(targets)}")
        print(f"batch size per GPU: {args.batch_size}")
        print(f"max batch char ratio: {args.max_batch_char_ratio}")
        print(f"speaker mode: {'fixed ' + configured_speakers[0] if args.speaker else 'random ' + ','.join(configured_speakers)}")
        if not args.speaker:
            for speaker, count in sorted(Counter(item['speaker'] for item in targets).items()):
                print(f"  speaker {speaker}: {count}")
        for source, count in sorted(source_target_counts.items()):
            print(f"  {source}: {count}")
        print(f"splits: {args.splits_dir} ratios={ratios}")
        return 0

    gpus: list[int] = []
    if not args.skip_synthesis and targets:
        gpus = resolve_gpus(args.gpus)
        print(f"synthesizing {len(targets)} rows on GPUs {gpus}")
        exitcodes, log_dir = run_synthesis(args, targets, gpus)
        failures = read_failure_count(log_dir)
        print(f"synthesis processes: exitcodes={exitcodes}; logged_failures={failures}; logs={log_dir}")
        if any(code != 0 for code in exitcodes):
            print("[warning] at least one synthesis worker exited non-zero", file=sys.stderr)

    filled = 0
    for item in targets:
        audio_path = Path(item["path"])
        if valid_wav(audio_path):
            records[int(item["row_index"])]["wav"] = item["manifest_path"]
            records[int(item["row_index"])]["tts_speaker"] = item["speaker"]
            filled += 1

    missing = [
        item for item in targets
        if not valid_wav(Path(item["path"]))
    ]
    print(f"collected valid synthesized WAVs: {filled}/{len(targets)}")

    if output_jsonl == input_path and args.backup:
        backup_path = args.backup_path.resolve() if args.backup_path else Path(str(input_path) + ".bak")
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"backup written: {backup_path}")
        else:
            print(f"backup kept: {backup_path}")

    atomic_write_jsonl(output_jsonl, records)
    print(f"updated JSONL written: {output_jsonl}")

    if missing:
        print(f"missing WAVs after synthesis: {len(missing)}", file=sys.stderr)
        if not args.allow_partial:
            print("not creating splits; rerun to retry missing rows or pass --allow-partial", file=sys.stderr)
            return 2
    if args.skip_split:
        print("split generation skipped")
        return 0

    eligible = [record for record in records if str(record.get("wav") or "").strip()]
    if not eligible:
        raise SystemExit("no non-empty wav rows available for splitting")
    min_counts = {
        "val": args.min_val_per_source,
        "test": args.min_test_per_source,
    }
    splits, metadata = source_stratified_split(records, ratios, args.split_seed, min_counts)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        atomic_write_jsonl(args.splits_dir / f"{name}.jsonl", splits[name])

    metadata.update(
        {
            "input": str(input_path),
            "updated_jsonl": str(output_jsonl),
            "rows_total": len(records),
            "rows_with_wav": len(eligible),
            "audio_dir": str(args.audio_dir),
            "gpus": gpus,
            "speaker_mode": "fixed" if args.speaker else "random",
            "speakers": list(configured_speakers),
            "speaker_seed": args.speaker_seed,
            "batch_size": args.batch_size,
            "max_batch_char_ratio": args.max_batch_char_ratio,
        }
    )
    (args.splits_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "splits written: "
        + ", ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
        + f"; metadata={args.splits_dir / 'split_metadata.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
