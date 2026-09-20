"""Build speaker-disjoint B0 manifests from extracted Emilia JSON/MP3 pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def collect(root: Path, lang: str) -> list[dict]:
    rows = []
    for sidecar in sorted(root.rglob("*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        audio = sidecar.with_suffix(".mp3")
        duration = float(record["duration"])
        speaker = str(record.get("speaker") or "").strip()
        identity = str(record.get("id") or "").strip()
        if not audio.is_file() or not speaker or not identity:
            raise ValueError(f"Missing MP3, speaker or id: {sidecar}")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration: {sidecar}")
        rows.append({"audio": str(audio.resolve()), "task": "audio_lm",
                     "source": "emilia", "lang": lang, "id": identity,
                     "speaker": speaker, "duration": duration,
                     "text": str(record.get("text") or "")})
    if not rows:
        raise ValueError(f"No Emilia JSON/MP3 pairs found: {root}")
    return rows


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    result = {name: [] for name in ("train", "dev", "test")}
    for lang in sorted({row["lang"] for row in rows}):
        groups: dict[str, list[dict]] = {}
        for row in rows:
            if row["lang"] == lang:
                groups.setdefault(row["speaker"], []).append(row)
        if len(groups) < 3:
            raise ValueError(f"Need at least three speakers for {lang}")
        remaining = sorted(groups)
        rng.shuffle(remaining)
        durations = {key: sum(row["duration"] for row in group)
                     for key, group in groups.items()}
        target = sum(durations.values()) * 0.01
        for name, reserve in (("dev", 2), ("test", 1)):
            selected = []
            total = 0.0
            for key in remaining:
                if len(remaining) - len(selected) <= reserve:
                    break
                if abs(total + durations[key] - target) < abs(total - target):
                    selected.append(key)
                    total += durations[key]
            if not selected:
                selected = [min(remaining, key=lambda key: abs(durations[key] - target))]
            for key in selected:
                result[name].extend(groups[key])
            chosen = set(selected)
            remaining = [key for key in remaining if key not in chosen]
        for key in remaining:
            result["train"].extend(groups[key])
    for items in result.values():
        rng.shuffle(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh-root", type=Path, required=True)
    parser.add_argument("--en-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/route_b/audio_lm_emilia"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    rows = collect(args.zh_root, "zh") + collect(args.en_root, "en")
    for label, keys in (("audio path", [row["audio"] for row in rows]),
                        ("language/id", [(row["lang"], row["id"]) for row in rows])):
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate {label}")
    splits = split_rows(rows, args.seed)
    metadata = {"seed": args.seed, "split_ratio": [0.98, 0.01, 0.01],
                "group_by": ["lang", "speaker"], "counts": {}}
    for name, items in splits.items():
        metadata["counts"][name] = {
            lang: {"rows": len(subset),
                   "hours": sum(row["duration"] for row in subset) / 3600,
                   "speakers": len({row["speaker"] for row in subset})}
            for lang in ("zh", "en")
            for subset in [[row for row in items if row["lang"] == lang]]
        }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, items in splits.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
