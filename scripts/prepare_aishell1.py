"""Create train/dev/test manifests and a character vocabulary for AISHELL-1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_transcripts(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = "".join(parts[1].split())
    return result


def collect_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/aishell1"))
    parser.add_argument("--output", type=Path, default=Path("data/aishell1/processed"))
    args = parser.parse_args()
    root = args.input / "data_aishell"
    transcripts = read_transcripts(root / "transcript/aishell_transcript_v0.8.txt")
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = sorted({char for text in transcripts.values() for char in text})
    (args.output / "vocab.txt").write_text("<blank>\n" + "\n".join(vocabulary) + "\n", encoding="utf-8")
    for subset in ("train", "dev", "test"):
        ids = collect_ids(root / f"resource_aishell/{subset}.txt")
        rows = []
        for utterance_id in ids:
            speaker, chapter, _ = utterance_id.split("_")
            audio = root / "wav" / speaker / chapter / f"{utterance_id}.wav"
            text = transcripts.get(utterance_id)
            if text and audio.exists():
                rows.append({"path": str(audio), "text": text})
        with (args.output / f"{subset}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{subset}_examples: {len(rows)}")
    print(f"vocab_size: {len(vocabulary) + 1}")
    print(f"processed_dir: {args.output}")


if __name__ == "__main__":
    main()
