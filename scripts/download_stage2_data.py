#!/usr/bin/env python3
"""Download the prepared Stage 2 speech-instruction corpus from ModelScope.

The repository contains the merged Stage 2 data used by Speech-MiniMind:
COIG-CQIA, COIG human-value, COIG translated, Firefly OpenQA/Dictionary,
and moss_speech_qa.  The current Chinese split on the 95 host does not include
VoiceAssistant-400K English rows.

The downloaded files are placed under ``data/speech2text_corpus/`` so that the
next step can run ``resample_stage2_mixed.py`` directly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_DATASET = "ghjghj1017/Speech-MiniMind"
DEFAULT_REVISION = "master"
DEFAULT_DOWNLOAD = Path("data/_stage2_modelscope")
DEFAULT_OUTPUT = Path("data/speech2text_corpus")


def snapshot_download(dataset_id: str, revision: str, destination: Path) -> Path:
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as error:
            raise SystemExit(
                "ModelScope is required. Run: python -m pip install modelscope"
            ) from error
        return Path(
            snapshot_download(
                dataset_id,
                repo_type="dataset",
                revision=revision,
                local_dir=str(destination),
            )
        )

    return Path(
        dataset_snapshot_download(
            dataset_id,
            revision=revision,
            local_dir=str(destination),
        )
    )


def find_named(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading prepared Stage 2 data: {args.dataset_id}")
    snapshot_root = snapshot_download(args.dataset_id, args.revision, args.download_dir)

    manifest = find_named(snapshot_root, "stage2_no_aishell.jsonl")
    audio = find_named(snapshot_root, "audio")
    splits = find_named(snapshot_root, "splits")
    if manifest is None and splits is None:
        raise SystemExit(
            f"No stage2_no_aishell.jsonl or splits directory found under {snapshot_root}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        copy_file(manifest, args.output / manifest.name)
    if audio is not None and audio.is_dir():
        copy_directory(audio, args.output / "audio")
    if splits is not None and splits.is_dir():
        copy_directory(splits, args.output / "splits")

    print(f"Stage 2 data is ready under {args.output}")
    print("Next: python scripts/resample_stage2_mixed.py --data "
          "data/speech2text_corpus/splits --splits train,val,test --sr 16000")


if __name__ == "__main__":
    main()
