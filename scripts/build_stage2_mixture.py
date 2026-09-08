"""Build a small, auditable stage-2 speech mixture.

The builder consumes JSONL files with ``audio/instruction/answer/task`` fields
and samples a fixed budget per source. AISHELL-1 manifests are converted to
transcription examples when optional external sources are absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_aishell(path: Path, limit: int, seed: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = []
    for row in rows[:limit]:
        text = "".join(row["text"].split())
        examples.append({
            "audio": row["path"],
            "instruction": "请将这段语音准确转写为中文文本。",
            "answer": text,
            "task": "transcription",
            "source": "aishell1",
        })
    return examples


def sample(items: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items[: min(count, len(items))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--sources", type=Path, default=Path("data/external_speech_instructions"))
    parser.add_argument("--output", type=Path, default=Path("data/stage2_mixture"))
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    quotas = {
        "asr": round(args.total * 0.50),
        "meeting": round(args.total * 0.20),
        "instruction": round(args.total * 0.20),
        "understanding": args.total - round(args.total * 0.50) - round(args.total * 0.20) - round(args.total * 0.20),
    }
    pools = {
        "asr": load_aishell(args.aishell / "train.csv", quotas["asr"], args.seed),
        "meeting": load_jsonl(args.sources / "meeting.jsonl"),
        "instruction": load_jsonl(args.sources / "instruction.jsonl"),
        "understanding": load_jsonl(args.sources / "understanding.jsonl"),
    }
    # If external sources are not available, keep the recipe runnable while
    # making provenance explicit. These fallback samples are still ASR data.
    for name in ("meeting", "instruction", "understanding"):
        if not pools[name]:
            pools[name] = load_aishell(args.aishell / "train.csv", quotas[name], args.seed + len(name))
            for item in pools[name]:
                item["source"] = "aishell1_fallback"
                item["task"] = "transcription_fallback"
    combined = []
    counts = {}
    for index, name in enumerate(("asr", "meeting", "instruction", "understanding")):
        chosen = sample(pools[name], quotas[name], args.seed + index)
        for item in chosen:
            item["mixture"] = name
        combined.extend(chosen)
        counts[name] = {"requested": quotas[name], "available": len(pools[name]), "selected": len(chosen)}
    random.Random(args.seed).shuffle(combined)
    train_count = int(len(combined) * 0.9)
    splits = {"train": combined[:train_count], "dev": combined[train_count:]}
    for split, rows in splits.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {"total_requested": args.total, "total_selected": len(combined), "seed": args.seed, "quotas": counts, "external_sources": str(args.sources), "warning": "Fallback groups use AISHELL-1 transcription when external JSONL files are absent."}
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
