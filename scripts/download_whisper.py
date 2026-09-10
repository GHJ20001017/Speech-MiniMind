"""Download the Whisper encoder weights for use as a frozen acoustic encoder.

OpenAI's Whisper-Small (Apache-2.0) is the recommended mature encoder to
replace the in-project Tiny Conformer once the latter's AISHELL-only
generalization becomes the bottleneck. This script fetches the full
Transformers-format model (weights + processor/tokenizer config) so it can be
loaded with ``WhisperModel.from_pretrained(...).encoder``.

By default it pulls from the **ModelScope mirror** (``openai-mirror/whisper-small``),
which is the preferred source for users in mainland China. If you prefer to fetch
straight from the upstream, pass ``--source huggingface`` (``openai/whisper-small``).

Usage
-----
.. code-block:: bash

    # ModelScope mirror (default, recommended in mainland China)
    python scripts/download_whisper.py --output outputs/whisper-small

    # Hugging Face upstream
    python scripts/download_whisper.py --source huggingface --output outputs/whisper-small

The output directory can then be referenced as ``--whisper-model outputs/whisper-small``
in the acoustic-encoder swap scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import huggingface_hub

HF_REPO = "openai/whisper-small"
MODELSCOPE_REPO = "openai-mirror/whisper-small"

# Default source: ModelScope is preferred (faster in mainland China), with the
# Hugging Face upstream as the fallback choice via --source huggingface.
DEFAULT_SOURCE = "modelscope"

# Files needed to run the frozen Whisper encoder (weights + config + processor).
INCLUDE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "normalizer.json",
    "special_tokens_map.json",
]


def download_hf(repo_id: str, target: Path) -> Path:
    """Snapshot-download from Hugging Face into ``target``."""
    return Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=INCLUDE_PATTERNS,
        )
    )


def download_modelscope(repo_id: str, target: Path) -> Path:
    """Snapshot-download from ModelScope into ``target``."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ModelScope 下载需要 modelscope 库: python -m pip install modelscope"
        ) from exc
    return Path(
        snapshot_download(
            model_id=repo_id,
            local_dir=str(target),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default=DEFAULT_SOURCE,
        help="download source; default modelscope（国内更快）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/whisper-small"),
        help="output directory for the downloaded model",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="override the repo id (defaults per --source)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.source == "huggingface":
        repo = args.repo or HF_REPO
        print(f"downloading from Hugging Face: {repo}")
        target = download_hf(repo, args.output)
    else:
        repo = args.repo or MODELSCOPE_REPO
        print(f"downloading from ModelScope: {repo}")
        target = download_modelscope(repo, args.output)

    print(f"model_root: {target}")
    print("usage: WhisperModel.from_pretrained('<model_root>').encoder  # 冻结用")


if __name__ == "__main__":
    main()