"""Download the FunASR Paraformer-zh-streaming model weights.

Paraformer-zh-streaming is Alibaba/FunASR's industrial real-time streaming
Chinese ASR model (Apache-2.0), a mature open-source substitute for the
in-project Tiny Conformer + CTC encoder. It is loaded through FunASR's
``AutoModel`` (not ``transformers``).

By default it pulls from the **ModelScope mirror**
(``iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online``),
which is the preferred source for users in mainland China. Pass ``--source
huggingface`` to fetch the upstream ``funasr/paraformer-zh-streaming`` instead.

Usage
-----
.. code-block:: bash

    python -m pip install funasr modelscope   # first time

    # ModelScope mirror (default, recommended in mainland China)
    python scripts/download_paraformer_streaming.py --output outputs/paraformer-streaming

    # Hugging Face upstream
    python scripts/download_paraformer_streaming.py --source huggingface --output outputs/paraformer-streaming

The output directory can then be referenced as ``AutoModel(model="<output>",
...).`` for use as the frozen acoustic front-end.

Note
----
This is a *complete* streaming ASR model. Unlike ``WhisperModel.from_pretrained(...).encoder``,
it does not expose the frame-level encoder hidden states as a simple attribute.
To use it as the frozen acoustic encoder feeding ``SpeechProjector``, you load it
with FunASR's ``AutoModel`` and need to reach inside for the encoder outputs (or
run it chunk-wise and capture encoder states), which is part of the acoustic-encoder
swap work — see README "换用成熟开源编码器".
"""

from __future__ import annotations

import argparse
from pathlib import Path

HF_REPO = "funasr/paraformer-zh-streaming"
MODELSCOPE_REPO = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"

# Default source: ModelScope is preferred (faster in mainland China), with the
# Hugging Face upstream as the fallback via --source huggingface.
DEFAULT_SOURCE = "modelscope"


def download_hf(repo_id: str, target: Path) -> Path:
    """Snapshot-download from Hugging Face into ``target``."""
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Hugging Face 下载需要 huggingface_hub 库: python -m pip install huggingface_hub"
        ) from exc
    return Path(
        huggingface_hub.snapshot_download(repo_id=repo_id, local_dir=target)
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
        default=Path("outputs/paraformer-streaming"),
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
    print("usage: from funasr import AutoModel; AutoModel(model='<model_root>', ...)")


if __name__ == "__main__":
    main()