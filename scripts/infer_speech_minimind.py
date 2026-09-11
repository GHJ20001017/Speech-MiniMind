"""CLI inference for the instruction-tuned Speech-MiniMind (chapter 04).

Loads the same pipeline as ``train_speech_minimind.py`` but in eval mode:

    audio ──▶ frozen acoustic encoder (conformer / paraformer)
              ──▶ SpeechProjector (frozen) ──▶ speech prefix embeddings
              ──concat──▶ MiniMind (tuned) ──▶ answer text

The tuned MiniMind checkpoint is the **chapter-04 output** (full or LoRA):

* full mode   -> ``outputs/04_speech_minimind_sft/model_epoch_XXX/``
* lora mode   -> ``outputs/04_speech_minimind_sft/lora_epoch_XXX/`` (needs peft
  to re-attach the adapters)

Usage
-----
.. code-block:: bash

    # full-mode tuned model, paraformer frontend (recommended)
    python scripts/infer_speech_minimind.py \\
      --audio path/to/utterance.wav \\
      --instruction "请将这段语音准确转写为中文文本。" \\
      --encoder-type paraformer \\
      --paraformer-model outputs/paraformer-streaming \\
      --projector-checkpoint outputs/03_speech_minimind_paraformer/projector_epoch_005.pt \\
      --minimind-model outputs/04_speech_minimind_sft/model_epoch_003

    # conformer frontend
    python scripts/infer_speech_minimind.py \\
      --audio path/to/utterance.wav \\
      --instruction "请将这段语音准确转写为中文文本。" \\
      --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \\
      --projector-checkpoint outputs/03_speech_minimind/projector_epoch_005.pt \\
      --minimind-model outputs/04_speech_minimind_sft/model_epoch_003

All weights are loaded from the paths you pass; nothing is downloaded here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.frozen_encoder import build_frozen_encoder  # noqa: E402
from model.minimind_adapter import generate_from_speech, load_minimind  # noqa: E402
from model.speech_projector import SpeechProjector  # noqa: E402
from scripts.analyze_audio import read_wav  # noqa: E402

SAMPLE_RATE = 16000


def resample_to_16k(audio: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """Resample a float32 waveform in [-1,1] to 16 kHz when needed."""
    if rate == SAMPLE_RATE:
        return audio, rate
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    new_len = round(audio.size * SAMPLE_RATE / rate)
    old = np.arange(audio.size) / rate
    new = np.arange(new_len) / SAMPLE_RATE
    return np.interp(new, old, audio).astype(np.float32), SAMPLE_RATE


def resolve_audio(value: str) -> tuple[np.ndarray, int]:
    """Read a WAV, resampling to 16 kHz when needed (paraformer needs 16 kHz)."""
    audio, rate = read_wav(Path(value))
    return resample_to_16k(audio, rate)


def load_pipeline(args, device: torch.device):
    """Load frozen encoder + projector + tuned MiniMind, all on ``device``."""
    encoder = build_frozen_encoder(
        args.encoder_type,
        checkpoint=str(args.encoder_checkpoint) if args.encoder_type == "conformer" else None,
        model_id=args.paraformer_model,
        device=device,
    )
    acoustic_dim = encoder.output_dim

    proj_ckpt = torch.load(args.projector_checkpoint, map_location=device, weights_only=False)
    llm_dim = int(proj_ckpt.get("llm_hidden_size", 768))
    projector = SpeechProjector(acoustic_dim, llm_dim).to(device)
    projector.load_state_dict(proj_ckpt["projector"])
    projector.eval()

    lm, tokenizer = load_minimind(args.minimind_model, device)
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()

    if args.tune == "lora":
        from peft import PeftModel  # re-attach LoRA adapters from the ckpt dir

        lm = PeftModel.from_pretrained(lm, args.minimind_model, is_trainable=False)
        lm.eval()

    return encoder, projector, lm, tokenizer, llm_dim


def run(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encoder, projector, lm, tokenizer, _ = load_pipeline(args, device)

    audio, rate = resolve_audio(args.audio)
    print(f"audio: {args.audio} ({audio.size / rate:.2f}s @ {rate} Hz)")

    waveforms = torch.from_numpy(audio[None].astype(np.float32))
    lengths = torch.tensor([audio.size], dtype=torch.long)
    with torch.no_grad():
        acoustic, acoustic_lengths = encoder.encode(waveforms, lengths, SAMPLE_RATE)
        projected = projector(acoustic)
        projected_lengths = projector.output_lengths(acoustic_lengths).clamp_max(projected.size(1))

    answer, token_ids = generate_from_speech(
        lm, tokenizer, projected, projected_lengths, args.instruction, device,
        max_new_tokens=args.max_new_tokens,
        max_speech_tokens=args.max_speech_tokens,
        temperature=args.temperature,
    )
    print("\n=== instruction ===")
    print(args.instruction)
    print("\n=== answer ===")
    print(answer if answer else "（空回答）")
    if args.verbose:
        print(f"\n=== generated token ids ({len(token_ids)}) ===")
        print(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, required=True, help="16-bit PCM WAV (16 kHz preferred; auto-resampled)")
    parser.add_argument("--instruction", default="请将这段语音准确转写为中文文本。", help="text instruction for the model")
    parser.add_argument("--encoder-type", choices=("conformer", "paraformer"), default="paraformer",
                        help="frozen acoustic encoder backend")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"),
                        help="conformer checkpoint (ignored for --encoder-type paraformer)")
    parser.add_argument("--paraformer-model", default=None, help="FunASR model id/dir (paraformer backend)")
    parser.add_argument("--projector-checkpoint", type=Path, required=True,
                        help="trained SpeechProjector ckpt (outputs/03_*/projector_epoch_XXX.pt)")
    parser.add_argument("--minimind-model", type=Path, required=True,
                        help="chapter-04 tuned MiniMind dir (full: model_epoch_XXX; lora: lora_epoch_XXX)")
    parser.add_argument("--tune", choices=("lora", "full"), default="full",
                        help="must match how the checkpoint was trained")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0, help=">0 enables sampling; 0 = greedy")
    parser.add_argument("--verbose", action="store_true", help="also print generated token ids")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()