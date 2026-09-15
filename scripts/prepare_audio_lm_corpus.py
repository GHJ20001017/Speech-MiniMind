"""Collect a pure-audio corpus manifest for Route-B B0 pre-training.

B0 teaches the audio LLM the *distribution* of codebook tokens before any
prompt/answer pairing exists, so it only needs a list of utterances.  This
script gathers every audio file we already have on disk and writes one JSONL
per split:

* ``data/aishell1/processed/{train,dev}.csv`` - Chinese read speech (16 kHz).
* ``data/moss_speech_qa/{train,dev}.jsonl`` - Qwen3-TTS question audio (24 kHz).
* ``data/voiceassistant400k_50k/{train,dev}.jsonl`` - English QA audio (deferred
  by default; pass ``--include-english`` to pull it in).

Output rows are ``{"audio": "<abs>", "source": ..., "lang": ..., "text": ...}``
and are consumed by ``scripts/cache_audio_tokens.py`` to produce the ``.npy``
shards the B0 trainer reads.  ``text`` is an optional transcript kept so the M0
reconstruction gate (``eval_codec_reconstruction.py``) can score round-trip
CER/WER directly on this manifest; it is ignored by training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/route_b/audio_lm"))
    parser.add_argument("--include-english", action="store_true",
                        help="also include the English voiceassistant400k audio")
    parser.add_argument("--include-moss", action="store_true", default=True,
                        help="include the moss Qwen3-TTS question audio (default on)")
    parser.add_argument("--no-include-moss", dest="include_moss", action="store_false")
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="cap rows per source per split (0 = no cap)")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_audio(value: str, manifest: Path) -> Path | None:
    """Resolve a manifest audio path, tolerating two different conventions.

    AISHELL CSVs store *repo-root-relative* paths (``data/aishell1/...``) while
    the moss/voiceassistant JSONLs store *manifest-relative* ones
    (``audio/000000.wav``).  Try the repo root first, then the manifest dir, and
    return ``None`` when neither exists so callers can skip quietly.

    The result is always absolutised: this manifest is consumed by scripts such
    as ``cache_audio_tokens.py`` that re-resolve relative entries against *their*
    manifest directory, so a relative path here would silently break.
    """
    path = Path(value)
    if path.is_absolute():
        return path if path.exists() else None
    for candidate in (ROOT / path, manifest.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return None


def collect_aishell(data_root: Path) -> dict[str, list[dict]]:
    splits = {"train": [], "dev": []}
    processed = data_root / "aishell1" / "processed"
    for split in ("train", "dev"):
        manifest = processed / f"{split}.csv"
        if not manifest.exists():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                audio = resolve_audio(row["path"], manifest)
                if audio is not None:
                    splits[split].append(
                        {"audio": str(audio), "source": "aishell1", "lang": "zh",
                         "text": (row.get("text") or "").strip()}
                    )
    return splits


def collect_jsonl(manifest: Path, source: str, lang: str) -> list[dict]:
    rows: list[dict] = []
    if not manifest.exists():
        return rows
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            audio = record.get("audio")
            if not audio:
                continue
            path = resolve_audio(audio, manifest)
            if path is not None:
                rows.append({"audio": str(path), "source": source, "lang": lang,
                             "text": _text_reference(record)})
    return rows


def _text_reference(record: dict) -> str:
    """Best-effort transcript of *the audio in this row*, for the M0 gate.

    Order matters: in the instruction corpora the ``audio`` field holds the
    **question/instruction** speech (VoiceAssistant-400K ``question_audio``,
    moss Qwen3-TTS synthesized instruction), so its spoken text is
    ``instruction`` - not ``answer``, which is the *text* reply and does not
    match the audio.  ``text`` covers AISHELL-style JSONL exports.
    """
    for key in ("instruction", "text", "transcript", "answer"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def main() -> None:
    args = parse_args()
    import random

    rng = random.Random(args.seed)
    data_root = args.data_root
    gathered: dict[str, list[dict]] = {"train": [], "dev": []}

    aishell = collect_aishell(data_root)
    for split in gathered:
        gathered[split].extend(aishell.get(split, []))

    if args.include_moss:
        moss_root = data_root / "moss_speech_qa"
        for split in gathered:
            gathered[split].extend(
                collect_jsonl(moss_root / f"{split}.jsonl", "moss_speech_qa", "zh")
            )

    if args.include_english:
        va_root = data_root / "voiceassistant400k_50k"
        for split in gathered:
            gathered[split].extend(
                collect_jsonl(va_root / f"{split}.jsonl", "voiceassistant400k_50k", "en")
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for split, rows in gathered.items():
        if args.max_per_source:
            by_source: dict[str, list[dict]] = {}
            for row in rows:
                by_source.setdefault(row["source"], []).append(row)
            capped: list[dict] = []
            for source, items in by_source.items():
                rng.shuffle(items)
                capped.extend(items[: args.max_per_source])
            rows = capped
        manifest = args.output / f"{split}.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        summary[split] = counts
        print(f"{split}: {len(rows)} rows ({counts}) -> {manifest}")

    (args.output / "metadata.json").write_text(
        json.dumps(
            {"data_root": str(data_root), "include_english": args.include_english,
             "include_moss": args.include_moss, "counts": summary},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
