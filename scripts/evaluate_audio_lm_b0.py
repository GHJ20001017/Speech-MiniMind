#!/usr/bin/env python3
"""Evaluate one B0 audio-LM checkpoint on five fixed dev code shards."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

from model.audio_codec import build_frozen_audio_codec
from model.audio_lm import AudioSample, build_audio_batch, build_vocab_spec, from_lm_tokens, to_lm_tokens
from model.qwen3_adapter import load_qwen3


def save_wav(codec, codes, path):
    codes = codes.long().contiguous()
    if codes.ndim != 2 or codes.shape[1] == 0:
        return 0.0
    wav, lengths = codec.decode(codes.unsqueeze(0), torch.tensor([codes.shape[1]], device=codes.device))
    n = int(lengths[0])
    sf.write(path, wav[0, :n].numpy(), codec.sample_rate, subtype="PCM_16")
    return n / codec.sample_rate


def constrained_argmax(logits, spec, slots):
    out = []
    for i in range(slots):
        q = i % spec.num_codebooks
        lo = spec.audio_offset + q * spec.codebook_size
        hi = lo + spec.codebook_size
        out.append(int(logits[i, lo:hi].argmax()) + lo)
    return torch.tensor(out, dtype=torch.long, device=logits.device)


@torch.inference_mode()
def cached_generate(model, prompt_ids, spec, max_new):
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    attn = torch.ones_like(ids)
    first = model(input_ids=ids, attention_mask=attn, use_cache=True)
    past = first.past_key_values
    logits = first.logits[:, -1, :]
    generated = []
    stopped = False
    for step in range(max_new):
        q = step % spec.num_codebooks
        lo = spec.audio_offset + q * spec.codebook_size
        hi = lo + spec.codebook_size
        next_logits = logits[0].clone()
        mask = torch.full_like(next_logits, float("-inf"))
        mask[lo:hi] = 0.0
        mask[spec.audio_eos_id] = 0.0
        next_id = int((next_logits + mask).argmax())
        if next_id == spec.audio_eos_id:
            stopped = True
            break
        if not spec.is_audio_token(next_id):
            stopped = True
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], device=device)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], dim=1)
        result = model(input_ids=token, attention_mask=attn, past_key_values=past, use_cache=True)
        past = result.past_key_values
        logits = result.logits[:, -1, :]
    return torch.tensor(generated, dtype=torch.long, device=device), stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--qwen3-model", required=True)
    ap.add_argument("--mimi-model", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--prefix-ratio", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, tok = load_qwen3(Path(args.checkpoint), device, dtype="bfloat16", prepare_for_speech=True)
    spec = build_vocab_spec(tok, 2048, 8)
    assert spec.audio_offset == 151672, spec
    codec = build_frozen_audio_codec("mimi", model_id=args.mimi_model, device=device, num_codebooks=8)
    files = sorted(Path(args.dev_dir).glob("*.npy"))[:5]
    if len(files) != 5: raise RuntimeError(f"expected 5 dev .npy files, found {len(files)}")
    rows = []
    for idx, f in enumerate(files):
        codes = np.load(f, allow_pickle=False)
        if codes.ndim == 1: codes = codes[None, :]
        if codes.shape[0] != spec.num_codebooks: raise ValueError(f"{f}: shape {codes.shape}")
        if codes.shape[1] < 4: raise ValueError(f"{f}: too short")
        if codes.min() < 0 or codes.max() >= spec.codebook_size: raise ValueError(f"{f}: invalid code")
        # Keep an explicit deterministic crop within the five-epoch training context.
        codes = torch.from_numpy(codes[:, :192]).long()
        sample_dir = out / f"sample_{idx:02d}"; sample_dir.mkdir(exist_ok=True)
        input_ids, labels, attn = build_audio_batch(tok, spec, [AudioSample(None, codes)], max_length=1552, max_answer_frames=192)
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
        result = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = result.logits[0, :-1]
        target = labels[0, 1:]
        mask = target >= spec.audio_offset
        target_audio = target[mask]
        pred_audio = logits[mask].argmax(dim=-1)
        n = int(mask.sum())
        ce = float(torch.nn.functional.cross_entropy(logits[mask].float(), target_audio).item())
        acc = float((pred_audio == target_audio).float().mean().item())
        # Exact supervised audio target excludes closing audio_end/eos; use first T*Q positions.
        tf_target = to_lm_tokens(codes.to(device), spec)
        tf_logits = logits[mask][: tf_target.numel()]
        tf_pred = constrained_argmax(tf_logits, spec, tf_target.numel())
        tf_codes = from_lm_tokens(tf_pred, spec).cpu()
        save_wav(codec, codes, sample_dir / "reference.wav")
        save_wav(codec, tf_codes, sample_dir / "teacher_forced.wav")
        prefix_frames = max(1, min(codes.shape[1] - 1, round(codes.shape[1] * args.prefix_ratio)))
        prefix = codes[:, :prefix_frames].to(device)
        prompt = [tok.bos_token_id, spec.audio_bos_id] if tok.bos_token_id is not None else [spec.audio_bos_id]
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        prompt_ids = torch.cat([prompt_ids, to_lm_tokens(prefix, spec).view(1, -1)], dim=1)
        gen, stopped = cached_generate(model, prompt_ids, spec, (codes.shape[1] - prefix_frames) * spec.num_codebooks)
        complete = (gen.numel() // spec.num_codebooks) * spec.num_codebooks
        gen = gen[:complete]
        if complete:
            gen_codes = from_lm_tokens(gen, spec).cpu()
            full_codes = torch.cat([prefix.cpu(), gen_codes], dim=1)
            save_wav(codec, full_codes, sample_dir / "free_running.wav")
            save_wav(codec, gen_codes, sample_dir / "free_running_continuation.wav")
        else:
            gen_codes = torch.empty((spec.num_codebooks, 0), dtype=torch.long)
        rows.append({"sample": idx, "file": str(f), "frames": int(codes.shape[1]), "prefix_frames": prefix_frames, "generated_tokens": int(gen.numel()), "generated_frames": int(gen_codes.shape[1]), "stopped": stopped, "teacher_forcing_ce": ce, "teacher_forcing_ppl": float(np.exp(ce)), "teacher_forcing_token_accuracy": acc, "supervised_audio_tokens": n})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / "metrics.json").write_text(json.dumps({"checkpoint": args.checkpoint, "device": str(device), "spec": vars(spec), "samples": rows}, indent=2, ensure_ascii=False))
    with (out / "metrics.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()
