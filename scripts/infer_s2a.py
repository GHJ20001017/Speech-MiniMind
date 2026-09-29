#!/usr/bin/env python3
"""Autoregressive S2A decoding for format-5 Thinker-Talker checkpoints.

The S2A checkpoint is trained on a complete text conversation and predicts the
selected assistant answer text and its delayed eight-codebook Mimi stream.
Teacher mode conditions on the reference answer; joint mode generates both
text and audio from history alone. Both use the training-aligned delayed
feedback rather than the text-only TTS decoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.audio_sampling import sample_audio, sample_text, GENERATION_CONFIG
from model.audio_codec import build_frozen_audio_codec
from scripts.evaluate_multitask_tts import save_wav
from trainer.train_audio_multitask import (
    AUDIO_CODEBOOK_DELAYS,
    AUDIO_STOP_ID,
    AUDIO_PAD_ID,
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    load_multitask_checkpoint,
    validate_init_checkpoint,
)
from model.qwen3_adapter import lm_head_module


def select_rows(data: Path, split: str, seed: int, limit: int) -> tuple[list[dict], Path]:
    from dataset.minimind_t2a_dataset import MiniMindT2ADataset

    dataset = MiniMindT2ADataset(data, task="s2a", split=split, seed=seed, limit=limit)
    return [dict(row) for row in dataset.rows], data


def _validate_messages(messages: object, answer: str) -> list[dict[str, str]]:
    if (not isinstance(messages, list) or len(messages) < 2
            or any(not isinstance(item, dict)
                   or item.get("role") not in {"system", "user", "assistant"}
                   or not isinstance(item.get("content"), str)
                   for item in messages)
            or messages[-1]["role"] != "assistant"
            or messages[-1]["content"] != answer):
        raise ValueError("selected S2A row does not contain a text conversation ending in its answer")
    if not any(item["role"] == "user" for item in messages[:-1]):
        raise ValueError("S2A conversation has no user context")
    return [dict(item) for item in messages]


def _answer_body_start(tokenizer, ids: list[int], assistant_start: int) -> int:
    """Training's bounded </think>\n\n onset search, without augmentation."""
    marker = list(tokenizer("</think>\n\n", add_special_tokens=False)["input_ids"])
    if not marker:
        raise ValueError("tokenizer returned an empty thinking-end marker")
    eos_text = getattr(tokenizer, "eos_token", None)
    eos_ids = (list(tokenizer(eos_text + "\n", add_special_tokens=False)["input_ids"])
               if eos_text else [tokenizer.eos_token_id])
    end = next((pos for pos in range(assistant_start, len(ids))
                if ids[pos:pos + len(eos_ids)] == eos_ids), len(ids))
    for position in range(assistant_start, min(assistant_start + 50, end)):
        if ids[position:position + len(marker)] == marker:
            return position + len(marker)
    return assistant_start


def _chat_prefix(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int], int]:
    """Return full ids, generation-prefix ids, and answer-body start."""
    prefix = list(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    ))
    full = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    ))
    if not prefix or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("S2A chat generation prefix is not a prefix of the full chat")
    return prefix, full, _answer_body_start(tokenizer, full, len(prefix))


@torch.inference_mode()
def synthesize_s2a(model, tokenizer, messages, max_frames: int, device):
    """Generate delayed audio while retaining the full prior text chat.

    The first assistant answer token is part of the initial context because the
    training alignment places the first q0 audio target one position later.
    Subsequent text positions are teacher-forced from the known answer text;
    audio code feedback is autoregressive and uses independently stopped streams.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    prefix, full, answer_start = _chat_prefix(tokenizer, messages)
    if len(full) > int(model.config.max_position_embeddings):
        raise ValueError("full teacher conversation exceeds model context; refusing to truncate")
    # Training starts audio targets at answer-body start + 1. Therefore the
    # initial forward includes the first answer token at row answer_start.
    initial_len = answer_start + 1
    if initial_len > len(full):
        raise ValueError("chat has no answer token after the assistant header")
    delay = max(AUDIO_CODEBOOK_DELAYS)
    available = int(model.config.max_position_embeddings) - initial_len - delay - 1
    if available < 1:
        raise ValueError("conversation leaves no room for delayed S2A audio and STOP")
    if max_frames > available:
        raise ValueError(
            f"requested {max_frames} frames exceed delayed S2A context capacity {available}"
        )

    inputs = torch.tensor([full[:initial_len]], dtype=torch.long, device=device)
    audio_inputs = torch.full(
        (1, initial_len, NUM_CODEBOOKS), AUDIO_PAD_ID, dtype=torch.long, device=device
    )
    attention = torch.ones_like(inputs)
    out = model.forward_streams(
        inputs, audio_inputs, attention_mask=attention, use_cache=True
    )
    past = out.past_key_values
    text_hidden, audio_hidden = out.text_hidden, out.audio_hidden

    streams = [[] for _ in range(NUM_CODEBOOKS)]
    sampled = [[] for _ in range(NUM_CODEBOOKS)]
    stopped = [False] * NUM_CODEBOOKS
    stop_steps: list[int | None] = [None] * NUM_CODEBOOKS
    vote_history: list[int] = []
    total_steps = max_frames + delay + 1
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id

    for step in range(total_steps):
        feedback = [AUDIO_PAD_ID] * NUM_CODEBOOKS
        for q, head in enumerate(model.audio_streams.heads):
            frame = step - AUDIO_CODEBOOK_DELAYS[q]
            if frame < 0:
                sampled[q].append(AUDIO_PAD_ID)
                continue
            predicted = sample_audio(head(audio_hidden[:, -1])[0], sampled[q])
            sampled[q].append(predicted)
            feedback[q] = predicted
            if predicted >= CODEBOOK_SIZE and not stopped[q]:
                stopped[q] = True
                stop_steps[q] = step
            elif not stopped[q] and frame < max_frames:
                streams[q].append((frame, predicted))
                feedback[q] = predicted
        vote_history.append(sum(stopped))
        if all(stopped):
            break
        if step + 1 >= total_steps:
            break

        next_position = initial_len + step
        text_id = full[next_position] if next_position < len(full) else pad
        next_input = torch.tensor([[text_id]], dtype=torch.long, device=device)
        next_audio = torch.tensor([[feedback]], dtype=torch.long, device=device)
        attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
        out = model.forward_streams(
            next_input, next_audio, attention_mask=attention,
            past_key_values=past, use_cache=True,
        )
        past = out.past_key_values
        text_hidden, audio_hidden = out.text_hidden, out.audio_hidden

    if not all(stopped):
        stop_reason = "max_new_frames" if len(vote_history) >= total_steps else "incomplete_stop"
    else:
        stop_reason = "audio_stop"
    if any(not values for values in streams):
        codes = torch.empty((NUM_CODEBOOKS, 0), dtype=torch.long)
    else:
        frame_count = min(len(values) for values in streams)
        codes = torch.tensor(
            [[code for _, code in values[:frame_count]] for values in streams], dtype=torch.long
        )
    return codes, {
        "generated_frames": int(codes.size(1)),
        "stop_votes": vote_history,
        "stop_steps": stop_steps,
        "stop_reason": stop_reason,
        "max_frames_used": max_frames,
        "answer_start": answer_start,
    }


@torch.inference_mode()
def synthesize_s2a_joint(model, tokenizer, history, max_frames: int, device,
                         max_text_tokens: int = 512):
    """Jointly autoregress text and delayed Mimi codes without answer leakage."""
    if max_frames < 1 or max_text_tokens < 1:
        raise ValueError("max_frames and max_text_tokens must be positive")
    if (not isinstance(history, list) or not history
            or any(not isinstance(item, dict)
                   or item.get("role") not in {"system", "user", "assistant"}
                   or not isinstance(item.get("content"), str) for item in history)
            or history[-1]["role"] == "assistant"):
        raise ValueError("joint history must contain text messages and omit the final assistant answer")
    prefix = list(tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True
    ))
    # Render only an empty assistant: its template-owned thinking scaffold is
    # deterministic and contains no reference answer tokens. Retain it rather
    # than applying training's random empty-thinking removal.
    empty = list(tokenizer.apply_chat_template(
        history + [{"role": "assistant", "content": ""}],
        tokenize=True, add_generation_prompt=False,
    ))
    if not prefix or len(empty) <= len(prefix) or empty[:len(prefix)] != prefix:
        raise ValueError("joint chat template generation prefix is not a prefix of empty assistant turn")
    trailer = empty[len(prefix):]
    im_end = getattr(tokenizer, "im_end_id", None)
    if im_end is None:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end < 0 or im_end not in trailer:
        raise ValueError("chat template did not expose an assistant im_end terminator")
    scaffold = trailer[:trailer.index(im_end)]
    body_start = _answer_body_start(tokenizer, empty, len(prefix))
    if body_start != len(prefix) + len(scaffold):
        raise ValueError("empty assistant scaffold does not end at the training answer-body onset")
    prefix = prefix + scaffold
    trailer = trailer[trailer.index(im_end):]
    if not trailer:
        raise ValueError("chat template assistant trailer is empty")

    delay = max(AUDIO_CODEBOOK_DELAYS)
    max_steps = max(max_text_tokens + len(trailer), max_frames + delay + 2)
    context_limit = int(model.config.max_position_embeddings)
    if len(prefix) + max_steps > context_limit:
        raise ValueError("joint S2A prefix and requested text/audio limits exceed model context")
    inputs = torch.tensor([prefix], dtype=torch.long, device=device)
    audio_inputs = torch.full((1, len(prefix), NUM_CODEBOOKS), AUDIO_PAD_ID, dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    out = model.forward_streams(
        inputs, audio_inputs, attention_mask=attention, use_cache=True
    )
    past = out.past_key_values
    text_hidden, audio_hidden = out.text_hidden, out.audio_hidden

    text_head = lm_head_module(model.thinker)
    generated = []
    streams = [[] for _ in range(NUM_CODEBOOKS)]
    sampled = [[] for _ in range(NUM_CODEBOOKS)]
    stopped = [False] * NUM_CODEBOOKS
    stop_steps = [None] * NUM_CODEBOOKS
    text_stopped = False
    text_reason = None
    pending_trailer = []
    audio_history = []
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    for step in range(max_steps):
        current_text = text_hidden[:, -1]
        current_audio = audio_hidden[:, -1]
        if pending_trailer:
            text_id = pending_trailer.pop(0)
        elif not text_stopped:
            text_id = sample_text(text_head(current_text)[0])
            if text_id == im_end:
                text_stopped = True
                text_reason = "im_end"
                pending_trailer = list(trailer[1:])
            else:
                generated.append(text_id)
                if len(generated) >= max_text_tokens:
                    text_stopped = True
                    text_reason = "max_text_tokens"
                    pending_trailer = list(trailer)
        else:
            text_id = pad

        feedback = [AUDIO_PAD_ID] * NUM_CODEBOOKS
        for q, head in enumerate(model.audio_streams.heads):
            # Step zero predicts only the first answer token. The first audio
            # code is predicted one step later, from that token's hidden state.
            frame = step - 1 - AUDIO_CODEBOOK_DELAYS[q]
            if frame < 0:
                sampled[q].append(AUDIO_PAD_ID)
                continue
            predicted = sample_audio(head(current_audio)[0], sampled[q])
            sampled[q].append(predicted)
            feedback[q] = predicted
            if predicted >= CODEBOOK_SIZE and not stopped[q]:
                stopped[q] = True
                stop_steps[q] = step - 1
            elif not stopped[q] and frame < max_frames:
                streams[q].append((frame, predicted))
                feedback[q] = predicted
        audio_history.append(sum(stopped))
        audio_done = all(stopped) or step >= max_frames + delay + 1
        if audio_done and text_stopped and not pending_trailer:
            break
        if step + 1 >= max_steps:
            break

        next_input = torch.tensor([[text_id]], dtype=torch.long, device=device)
        next_audio = torch.tensor([[feedback]], dtype=torch.long, device=device)
        attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
        out = model.forward_streams(
            next_input, next_audio, attention_mask=attention,
            past_key_values=past, use_cache=True,
        )
        past = out.past_key_values
        text_hidden, audio_hidden = out.text_hidden, out.audio_hidden

    if any(not values for values in streams):
        codes = torch.empty((NUM_CODEBOOKS, 0), dtype=torch.long)
    else:
        frame_count = min(len(values) for values in streams)
        codes = torch.tensor([[code for _, code in values[:frame_count]]
                              for values in streams], dtype=torch.long)
    if all(stopped):
        audio_reason = "audio_stop"
    else:
        audio_reason = "max_new_frames"
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return codes, {
        "generated_text": generated_text,
        "text_stop": text_stopped,
        "text_stopped": text_stopped,
        "text_stop_reason": text_reason,
        "audio_stop_reason": audio_reason,
        "generated_text_tokens": len(generated),
        "generated_frames": int(codes.size(1)),
        "stop_votes": audio_history,
        "stop_steps": stop_steps,
        "max_frames_used": max_frames,
        "max_text_tokens": max_text_tokens,
        "answer_start": len(prefix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=False, help="MiniMind-O conversation parquet")
    parser.add_argument("--prompts-json", type=Path, help="JSON list of histories without final assistant answers")
    parser.add_argument("--mode", choices=("teacher", "joint"), default="joint")
    parser.add_argument("--mimi-model", required=True, help="Mimi decoder matching the parquet's 24 kHz/8x2048 format")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_samples < 1 or args.max_new_frames < 1:
        parser.error("num-samples and max-new-frames must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    metadata = validate_init_checkpoint(args.checkpoint)
    if metadata.get("task") != "s2a":
        parser.error("checkpoint metadata task must be s2a")
    if args.prompts_json is not None and args.data is not None:
        parser.error("--data and --prompts-json are mutually exclusive")
    if args.prompts_json is None and args.data is None:
        parser.error("one of --data or --prompts-json is required")
    if args.prompts_json is not None and args.mode != "joint":
        parser.error("--prompts-json requires --mode joint")
    if args.mode == "joint" and args.prompts_json is not None:
        prompt_histories = json.loads(args.prompts_json.read_text(encoding="utf-8"))
        if not isinstance(prompt_histories, list):
            parser.error("--prompts-json must contain a JSON list of histories")
        rows = [{"messages": history} for history in prompt_histories]
        manifest = args.prompts_json
    else:
        rows, manifest = select_rows(args.data, args.split, args.seed, args.num_samples)
    if len(rows) > args.num_samples:
        rows = rows[:args.num_samples]
    if not rows:
        raise ValueError("selected S2A split is empty")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer = load_multitask_checkpoint(args.checkpoint, device)
    codec = build_frozen_audio_codec("mimi", model_id=args.mimi_model, device=device, num_codebooks=NUM_CODEBOOKS)
    args.output.mkdir(parents=True)
    report = {
        "generation": GENERATION_CONFIG, "seed": args.seed,
        "checkpoint": str(args.checkpoint), "data": str(manifest), "split": args.split,
        "device": str(device), "decoder": "s2a_joint" if args.mode == "joint" else "s2a_delayed_autoregressive_text_teacher_forced",
        "audio_codebook_delays": list(AUDIO_CODEBOOK_DELAYS),
        "max_new_frames": args.max_new_frames, "samples": [],
    }
    for index, row in enumerate(rows):
        if args.mode == "joint":
            messages = row["messages"] if args.prompts_json is not None else row["messages"][:-1]
            codes, metrics = synthesize_s2a_joint(
                model, tokenizer, messages, args.max_new_frames, device
            )
            answer_text = metrics["generated_text"]
        else:
            messages = _validate_messages(row["messages"], row["text"])
            codes, metrics = synthesize_s2a(model, tokenizer, messages, args.max_new_frames, device)
            answer_text = row["text"]
        sample_dir = args.output / f"sample_{index:02d}"
        sample_dir.mkdir()
        (sample_dir / "conversation.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        np.save(sample_dir / "generated_codes.npy", codes.numpy())
        metrics.update({"sample": sample_dir.name, "answer_text": answer_text, "generated_wav": None})
        if codes.size(1):
            metrics["generated_seconds"] = save_wav(codec, codes, sample_dir / "generated.wav")
            metrics["generated_wav"] = str(sample_dir / "generated.wav")
        else:
            metrics["warning"] = "no complete audio frame was generated"
        report["samples"].append(metrics)
        (args.output / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"Generated S2A samples are under {args.output}")


if __name__ == "__main__":
    main()
