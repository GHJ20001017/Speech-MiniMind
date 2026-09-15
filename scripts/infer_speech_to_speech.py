"""Inference: speech in -> codes -> audio-native LLM -> codes -> speech out.

Loads a Route-B checkpoint (B0 or B1/B2), encodes an input WAV with the frozen
codec, and autoregressively generates the answer codes, which the codec decodes
back to a waveform.

Usage::

    python scripts/infer_speech_to_speech.py \
        --audio path/to/question.wav \
        --model outputs/06_route_b_s2s/model_epoch_003 \
        --codec-type mimi --device cuda:0 \
        --output outputs/route_b_answer.wav

The script prints the token count / duration ratio and, when ``--print-hypothesis``
is set, runs the generated audio through the frozen SenseVoice frontend for a
quick intelligibility check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from model.audio_codec import build_frozen_audio_codec  # noqa: E402
from model.audio_lm import from_lm_tokens, generate_audio_tokens  # noqa: E402
from model.minimind_adapter import load_minimind  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="input question WAV")
    parser.add_argument("--model", type=Path, required=True,
                        help="Route-B checkpoint dir (contains the tokenizer with audio tokens)")
    parser.add_argument("--codec-type", default="mimi", choices=("mimi", "encodec"))
    parser.add_argument("--codec-model", default=None)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-frames", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-prompt-frames", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("outputs/route_b_answer.wav"))
    parser.add_argument("--print-hypothesis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def audio_offset_for(model_path: Path) -> int:
    """Read the audio block offset recorded by the trainer, if available."""
    for candidate in (model_path / "config.json", model_path.parent / "config.json"):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "audio_offset" in data:
                return int(data["audio_offset"])
    return -1  # unknown; derive from tokenizer instead


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    import soundfile as sf

    model, tokenizer = load_minimind(args.model, device)
    model.eval()

    codec = build_frozen_audio_codec(args.codec_type, args.codec_model, device)
    codebook_size = codec.codebook_size
    num_codebooks = codec.num_codebooks

    from model.audio_lm import AudioVocabSpec

    offset = audio_offset_for(args.model)
    if offset < 0:
        # Special tokens were appended last, so the block starts here.
        offset = len(tokenizer.get_vocab()) - 3 - codebook_size * num_codebooks
    spec = AudioVocabSpec(
        text_vocab_size=offset,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        audio_offset=offset,
        audio_bos_id=tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        audio_eos_id=tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        audio_pad_id=tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
    )
    print(f"vocab: text={spec.text_vocab_size} audio={spec.audio_vocab_size} "
          f"total={spec.total_vocab_size}")

    waveform, rate = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    tensor = torch.from_numpy(waveform.astype(np.float32))[None, :]
    lengths = torch.tensor([tensor.size(1)], dtype=torch.long)
    codes, code_lengths = codec.encode(tensor, lengths, rate)
    prompt_frames = min(int(code_lengths[0]), args.max_prompt_frames)
    prompt_codes = codes[0, :, :prompt_frames]
    print(f"input: {len(waveform) / rate:.2f}s -> {prompt_frames} code frames")

    prompt_ids = [spec.audio_bos_id]
    from model.audio_lm import to_lm_tokens

    prompt_ids.extend(to_lm_tokens(prompt_codes, spec).tolist())
    prompt_ids.append(spec.audio_eos_id)
    prompt_ids.append(spec.audio_bos_id)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = generate_audio_tokens(
        model, spec, prompt_tensor,
        max_new_tokens=args.max_new_frames * num_codebooks,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=spec.audio_eos_id,
    )
    if not generated:
        raise SystemExit("model produced no audio tokens")
    # Trim to whole frames.
    generated = generated[: (len(generated) // num_codebooks) * num_codebooks]
    frames = torch.tensor(generated, dtype=torch.long)
    answer_codes = from_lm_tokens(frames, spec)[None, :, :]
    print(f"generated: {answer_codes.size(2)} code frames "
          f"(~{answer_codes.size(2) / codec.frame_rate_hz:.2f}s)")

    recon, recon_lengths = codec.decode(answer_codes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), recon[0, : int(recon_lengths[0])].numpy(), codec.sample_rate,
             subtype="PCM_16")
    print(f"wrote: {args.output}")

    if args.print_hypothesis:
        try:
            from funasr import AutoModel
            import re

            asr = AutoModel(model=args.sensevoice_model, device=args.device, disable_update=True)
            result = asr.generate(input=recon[0, : int(recon_lengths[0])].numpy(), cache={},
                                  language="zh", use_itn=False, batch_size_s=60)
            text = result[0]["text"] if result else ""
            print("ASR:", re.sub(r"<\|[^|]*\|>", "", text).strip())
        except Exception as error:  # noqa: BLE001 - optional dependency
            print(f"hypothesis skipped ({error})")


if __name__ == "__main__":
    main()
