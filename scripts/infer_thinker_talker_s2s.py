#!/usr/bin/env python3
"""Waveform -> frozen SenseVoice -> conditioned Thinker -> text + Talker Mimi waveform."""
import argparse
# Local package imports below intentionally follow the direct-script path setup.
# ruff: noqa: E402
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from dataset.thinker_talker_s2s import speech_prompt, load_waveform
from model.audio_codec import build_frozen_audio_codec
from model.speech_input import load_s2s_checkpoint
from scripts.infer_s2a import synthesize_s2a_joint
from scripts.evaluate_multitask_tts import save_wav


@torch.inference_mode()
def synthesize_s2s(model, tokenizer, waveform, max_frames, device, max_text_tokens=512):
    question = model.encode_waveforms([waveform])[0]
    history, _, _, slots = speech_prompt(tokenizer, question.shape[0])
    return synthesize_s2a_joint(
        model, tokenizer, history, max_frames, device, max_text_tokens=max_text_tokens,
        pad_stopped_streams=True,
        speech_inputs={"speech_features": [question], "speech_mask": slots.unsqueeze(0).to(device)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=None, help="override local SenseVoice directory")
    parser.add_argument("--mimi-model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory for answer.wav and answer.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-frames", type=int, default=500)
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output exists; choose a new directory")
    if min(args.max_new_frames, args.max_text_tokens) < 1:
        parser.error("generation limits must be positive")
    waveform = load_waveform(args.audio)
    torch.manual_seed(args.seed)
    model, tokenizer = load_s2s_checkpoint(args.checkpoint, args.device, encoder_path=args.encoder)
    codec = build_frozen_audio_codec("mimi", model_id=args.mimi_model,
                                     device=args.device, num_codebooks=8)
    if codec.num_codebooks != 8 or codec.codebook_size != 2048:
        raise ValueError("S2S requires Mimi 8x2048")
    generated, report = synthesize_s2s(model, tokenizer, waveform,
                                      args.max_new_frames, args.device, args.max_text_tokens)
    if generated.shape[1] == 0:
        raise RuntimeError("model produced no complete audio frames; frontend quality is unvalidated")
    args.output.mkdir(parents=True, exist_ok=False)
    report["duration_seconds"] = save_wav(codec, generated, args.output / "answer.wav")
    report.update(checkpoint=str(args.checkpoint), input_audio=str(args.audio), seed=args.seed)
    (args.output / "answer.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["generated_text"])


if __name__ == "__main__":
    main()
