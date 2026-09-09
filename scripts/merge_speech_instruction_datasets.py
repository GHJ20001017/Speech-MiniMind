"""Merge the three stage-2 speech instruction datasets into one standard corpus.

Sources (all expected under the project ``data/`` dir on the remote host):
  - ``speech_instructions/{train,dev}.jsonl``   (AISHELL-1 transcription, zh)
  - ``moss_speech_qa/{train,dev}.jsonl``        (moss-003 SFT speech QA, zh)
  - ``voiceassistant400k_50k/{train,dev}.jsonl``(VoiceAssistant-400K speech QA, en)

Each source uses a *different* audio-path base, so this script resolves every
row's ``audio`` to an **absolute path** before writing the merged corpus. That
keeps the manifest valid from anywhere and avoids copying/symlinking gigbytes of
audio (the training loader treats absolute paths verbatim).

Every merged row uses the project standard schema:
  ``audio, instruction, answer, task, source, lang``

moss ``history`` (multi-turn) is intentionally dropped — the merged corpus is
single-turn only, matching ``InstructionJSONLDataset``.

Merged output: ``--output/{train,dev}.jsonl`` + ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# source key -> (subdir, lang)
SOURCES: dict[str, tuple[str, str]] = {
    "speech_instructions": ("speech_instructions", "zh"),
    "moss_speech_qa": ("moss_speech_qa", "zh"),
    "voiceassistant400k_50k": ("voiceassistant400k_50k", "en"),
}


# source key -> (subdir, lang, audio_base)
# audio_base is the directory the row's relative `audio` path is resolved against:
#   - speech_instructions  stores paths relative to the PROJECT ROOT (.. of data-root)
#   - moss / voiceassistant store paths relative to their own subdir
DATA_ROOT = None  # set in main

def audio_base_for(key: str, data_root: Path) -> Path:
    if key == "speech_instructions":
        return data_root.parent  # project root
    return data_root / key


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="project data dir containing the three source subdirs")
    parser.add_argument("--output", type=Path, default=Path("data/stage2_mixed"),
                        help="output dir for merged {train,dev}.jsonl + metadata.json")
    parser.add_argument("--skip-missing-audio", action="store_true",
                        help="drop rows whose audio file does not exist on disk")
    args = parser.parse_args()
    DATA_ROOT = args.data_root

    args.output.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[dict]] = {"train": [], "dev": []}
    counts: dict[str, dict[str, int]] = {}

    for key, (subdir, lang) in SOURCES.items():
        src_dir = args.data_root / subdir
        audio_base = audio_base_for(key, args.data_root)
        counts[key] = {}
        for split in ("train", "dev"):
            rows = load_rows(src_dir / f"{split}.jsonl")
            out = []
            for row in rows:
                audio = str(row.get("audio", "")).strip()
                instruction = str(row.get("instruction", "")).strip()
                answer = str(row.get("answer", "")).strip()
                if not audio or not answer:
                    continue
                audio_path = Path(audio)
                if not audio_path.is_absolute():
                    audio_path = audio_base / audio_path
                if args.skip_missing_audio and not audio_path.exists():
                    continue
                out.append({
                    "audio": str(audio_path.resolve()),
                    "instruction": instruction or "请描述这段语音的内容。",
                    "answer": answer,
                    "task": str(row.get("task", "")).strip() or key,
                    "source": str(row.get("source", "")).strip() or key,
                    "lang": lang,
                })
            merged[split].extend(out)
            counts[key][split] = len(out)
            print(f"{key}/{split}: {len(out)} rows")

    for split, rows in merged.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    totals = {split: len(rows) for split, rows in merged.items()}
    by_lang = {}
    for split, rows in merged.items():
        lang_counts = {}
        for row in rows:
            lang_counts[row["lang"]] = lang_counts.get(row["lang"], 0) + 1
        by_lang[split] = lang_counts

    metadata = {
        "source_counts": counts,
        "totals": totals,
        "by_lang": by_lang,
        "schema": ["audio", "instruction", "answer", "task", "source", "lang"],
        "note": "audio paths are absolute; moss history dropped (single-turn).",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nmerged totals:", json.dumps(metadata["totals"], ensure_ascii=False))
    print("by_lang:", json.dumps(by_lang, ensure_ascii=False))
    print(f"written to: {args.output}")


if __name__ == "__main__":
    main()