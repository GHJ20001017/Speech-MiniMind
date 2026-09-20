"""Instruction-tune a Qwen3-0.6B Speech LLM on speech instruction data.

Pipeline:

    audio ──> TinyConformer.encoder (frozen) ──> SpeechProjector
                ──> speech prefix embeddings
                ──concat──> Qwen3-0.6B (LoRA or full fine-tuned) ──> answer text

The acoustic encoder is always frozen. The SpeechProjector is frozen by default
and can optionally be fine-tuned with ``--tune-projector`` when the Qwen3
backbone and speech frontend need to adapt together.

Data: a stage-2 directory of ``wav, answer`` manifests (``train.jsonl`` and
``dev.jsonl`` / ``val.jsonl``) as produced by
``scripts/synthesize_stage2_tts_and_split.py``. Rows carry only the audio path
and the reference answer; every sample is trained against a fixed system prompt
(``--system-prompt``, default "你是一个语音助手，根据用户的音频内容回答用户的问题").
The train/dev split comes from the manifests, so no re-splitting happens here.
The optional ``lang`` field can filter rows via ``--lang-filter``.

Samples are laid out with Qwen3's non-thinking chat template, the spoken
utterance sitting inside the ``user`` turn::

    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>{语音}<|audio_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>

The empty ``<think>\n\n</think>\n\n`` block is how Qwen3 disables thinking - it is
already closed, so the model answers straight away instead of opening a reasoning
block. ``model/chat_format.py`` owns this layout and asserts at load time that it
still matches the tokenizer's own template.

Loss is computed only over the ``{answer}<|im_end|>`` span; the system/user
prefix, the speech embeddings, both audio markers and the assistant header are
all masked.

Two things make the reported loss usable: ``train/loss_step`` is the cross-entropy
averaged over **every rank's** batch rather than rank 0's alone, and
``train/loss_ema`` smooths it with ``--loss-ema``. One ``--batch-size`` batch is a
noisy estimate when answers run from a few characters to a thousand - measured on
the stage-2 corpus the per-sample loss spans roughly 2.2 to 9.8, so a 4-sample
batch jumps by about +-0.7 with a flat mean. ``train/lr`` records the learning
rate actually in use. ``--lr-schedule cosine`` (default) applies a linear warmup
over ``--warmup-ratio`` of all steps followed by a cosine decay to
``--min-lr-ratio`` of the base lr; ``--lr-schedule none`` keeps the lr flat.

Note on the audio markers: ``<|audio_start|>`` / ``<|audio_end|>`` are appended
to Qwen3's tokenizer by ``model/chat_format.py`` and get a neutral initial
embedding. LoRA adapters do not cover ``embed_tokens``, so under ``--tune lora``
those two rows stay at that initial value for the whole run; ``--tune full``
trains them with everything else.

Memory: the loss goes through ``model/chunked_loss.py`` - the LM head is applied
``--loss-chunk`` positions at a time and recomputed in the backward pass, instead
of materialising the full ``(batch, length, 151672)`` logits four times over.
That one-shot loss is the largest single allocation in a ``--tune full`` run
(~10 GiB at ``--batch-size 4`` and ``--max-length 2048``).

Requires: peft (``pip install peft``) for ``--tune lora``, plus the same
torch/transformers as the rest of the project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Fragmentation, not the model size, is what turns a tight run into an OOM. Set
# before torch is imported, and only as a default: an explicit user setting wins.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.speech_dataset import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    SpeechAugmentConfig,
    SpeechInstructionDataset,
)
from model.frozen_encoder import build_frozen_encoder  # noqa: E402
from model import ddp_utils  # noqa: E402
from model.chat_format import encode_answer, encode_assistant_header, encode_prompt  # noqa: E402
from model.chunked_loss import chunked_cross_entropy  # noqa: E402
from model.qwen3_adapter import (  # noqa: E402
    forward_hidden_states,
    lm_head_module,
    load_qwen3,
    token_embeddings,
)
from model.speech_projector import SpeechProjector  # noqa: E402

SAMPLE_RATE = 16000

# Full fine-tuning a 0.6B model needs a far smaller step than LoRA does; spending
# the wrong one either diverges (full) or barely moves (lora). Resolved from
# --tune when --lr is not given.
DEFAULT_LR = {"lora": 2e-4, "full": 2e-5}

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None


def collate(batch: list[tuple[torch.Tensor, int, str, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    waveforms, rates, instructions, answers = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in waveforms], dtype=torch.long)
    return pad_sequence(waveforms, batch_first=True), lengths, list(instructions), list(answers)


def make_sft_batch(
    model,
    tokenizer,
    projected: torch.Tensor,
    projected_lengths: torch.Tensor,
    instructions: list[str],
    answers: list[str],
    device: torch.device,
    max_length: int,
    max_speech_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, attention_mask, labels) with prompt masking.

    Layout per sample (Qwen3 non-thinking chat template, speech inside the ``user`` turn)::

        <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n<|audio_start|>
        [speech tokens]
        <|audio_end|><|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>

    Labels mark only the trailing ``{answer}<|im_end|>`` as trainable; the
    system/user prefix, the speech embeddings, both audio markers, the assistant
    header and the empty think block are all masked with -100.
    """
    seqs_emb: list[torch.Tensor] = []
    seqs_mask: list[torch.Tensor] = []
    seqs_label: list[torch.Tensor] = []
    for speech, speech_len, instruction, answer in zip(
        projected, projected_lengths.tolist(), instructions, answers
    ):
        prefix = torch.tensor(
            encode_prompt(tokenizer, instruction), dtype=torch.long, device=device
        )
        header = torch.tensor(
            encode_assistant_header(tokenizer), dtype=torch.long, device=device
        )
        answer_ids = torch.tensor(
            encode_answer(tokenizer, answer), dtype=torch.long, device=device
        )

        # Keep the whole prefix/header plus at least one answer token; the
        # speech prefix absorbs whatever budget is left.
        text_budget = max_length - prefix.numel() - header.numel() - 1
        speech = speech[: min(speech_len, max_speech_tokens, max(1, text_budget))]
        suffix = torch.cat([header, answer_ids])[
            : max_length - prefix.numel() - speech.size(0)
        ]

        # `model` may be a PeftModel (optionally DDP-wrapped); token_embeddings
        # unwraps both to reach the base LM's embedding table. The projector
        # keeps its own dtype, so the speech block is cast at the handoff.
        prefix_embeds = token_embeddings(model, prefix).unsqueeze(0)  # [1, P, H]
        suffix_embeds = token_embeddings(model, suffix).unsqueeze(0)  # [1, Q, H]

        emb = torch.cat(
            [prefix_embeds, speech.to(prefix_embeds.dtype).unsqueeze(0), suffix_embeds],
            dim=1,
        )  # [1, P+S+Q, H]
        seq_len = emb.size(1)
        pad_len = max_length - seq_len
        if pad_len > 0:
            emb = torch.cat([emb, torch.zeros(1, pad_len, emb.size(2), device=device)], dim=1)
        mask = torch.zeros(max_length, device=device, dtype=torch.long)
        mask[:seq_len] = 1
        label = torch.full((max_length,), -100, device=device, dtype=torch.long)
        answer_start = prefix.numel() + speech.size(0) + header.numel()
        label[answer_start:seq_len] = suffix[header.numel():]

        seqs_emb.append(emb)
        seqs_mask.append(mask)
        seqs_label.append(label)

    return torch.cat(seqs_emb, dim=0), torch.stack(seqs_mask), torch.stack(seqs_label)


def run_epoch(args, encoder, projector, lm, tokenizer, loader, device, optimizer=None,
              sampler=None, epoch=0, scheduler=None, loss_ema=None):
    """Run one pass over ``loader``; returns ``(mean_loss, loss_ema)``.

    ``loss_ema`` is only maintained while training (``optimizer`` is not None), so
    dev calls pass ``None`` and ignore the second element of the result.
    """
    training = optimizer is not None
    if sampler is not None:
        ddp_utils.set_epoch(sampler, epoch)
    projector.train(training and args.tune_projector)
    encoder.train(False)
    lm.train(training)
    projector_module = ddp_utils.unwrap(projector)
    total_loss = 0.0
    steps = 0
    for waveforms, lengths, instructions, answers in tqdm(loader, desc="train" if training else "dev", unit="batch", disable=not ddp_utils.is_main()):
        with torch.no_grad():
            acoustic, acoustic_lengths = encoder.encode(
                waveforms.to(device), lengths.to(device), SAMPLE_RATE,
                feature_augment=args.augment_mel and training,
            )
        projected = projector(acoustic)
        projected_lengths = projector_module.output_lengths(acoustic_lengths).clamp_max(projected.size(1))

        inputs_embeds, attention_mask, labels = make_sft_batch(
            lm, tokenizer, projected, projected_lengths, instructions, answers,
            device, args.max_length, args.max_speech_tokens,
        )
        with torch.set_grad_enabled(training):
            # Decoder first, LM head second: the head is applied (and recomputed
            # in backward) chunk by chunk, so the (B, L, 152k) logits never have
            # to exist in full - see model/chunked_loss.py.
            hidden = forward_hidden_states(lm, attention_mask, inputs_embeds=inputs_embeds)
            loss, _ = chunked_cross_entropy(
                lm_head_module(lm), hidden, labels, chunk_size=args.loss_chunk,
            )
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(lm.parameters()) + list(projector.parameters()) if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step_loss = ddp_utils.all_reduce_mean(loss.detach().item())
            loss_ema = step_loss if loss_ema is None else (
                (1.0 - args.loss_ema) * loss_ema + args.loss_ema * step_loss
            )
            if args.wandb and ddp_utils.is_main():
                wandb.log({
                    "train/loss_step": step_loss,
                    "train/loss_ema": loss_ema,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })
        total_loss += loss.detach().item()
        steps += 1
    return ddp_utils.all_reduce_mean(total_loss / max(steps, 1)), loss_ema


def add_lora(
    model,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str] | None = None,
):
    """Apply LoRA low-rank adapters to Qwen3's attention projections in place.

    Uses peft. Raises if peft is missing. Note that LoRA does not touch
    ``embed_tokens``, so the audio marker rows keep their initial value.
    """
    try:
        from peft import LoraConfig, get_peft_model

        targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(
            f"peft: trainable={trainable:,} "
            f"({100 * trainable / sum(p.numel() for p in peft_model.parameters()):.2f}%)"
        )
        return peft_model
    except ImportError:
        raise SystemExit(
            "peft not installed. Run: pip install peft  (needed for LoRA SFT)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="stage-2 split dir with train.jsonl (+ dev.jsonl/val.jsonl)")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--encoder-type", choices=("sensevoice", "conformer", "paraformer"), default="sensevoice",
                        help="frozen acoustic encoder backend (sensevoice=batched offline default)")
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall",
                        help="SenseVoice-Small model id or local directory")
    parser.add_argument("--paraformer-model", default=None,
                        help="FunASR model id/dir for --encoder-type paraformer")
    parser.add_argument("--projector-checkpoint", type=Path, required=True,
                        help="trained SpeechProjector ckpt (outputs/03_speech_qwen3_projector/projector_epoch_XXX.pt)")
    parser.add_argument("--tune-projector", action=argparse.BooleanOptionalAction, default=False,
                        help="also fine-tune SpeechProjector; default keeps the frontend frozen")
    parser.add_argument("--projector-lr", type=float, default=None,
                        help="Projector learning rate when --tune-projector (default: --lr)")
    parser.add_argument("--qwen3-model", type=Path, required=True,
                        help="local Qwen3-0.6B dir (or a checkpoint previously tuned by this script)")
    parser.add_argument("--llm-dtype", default="auto",
                        choices=("auto", "bfloat16", "float16", "float32"),
                        help="backbone dtype. 'auto' is transformers' default: float32 master "
                             "weights even for a bfloat16 checkpoint, i.e. the safest thing to "
                             "full fine-tune at 2x backbone memory; 'bfloat16'/'float16' halve "
                             "that when the backbone stays frozen. Speech embeddings are cast "
                             "to the backbone dtype either way")
    parser.add_argument("--tune", choices=("lora", "full"), default="lora",
                        help="lora: LoRA adapters only (~0.5%% trainable, needs peft); "
                             "full: full-parameter LLM fine-tune (all LLM weights trainable)")
    parser.add_argument("--output", type=Path, default=Path("outputs/04_speech_qwen3_sft"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for parallel audio loading/augmentation")
    parser.add_argument("--lr", type=float, default=None,
                        help="LLM learning rate (default: 2e-4 for --tune lora, 2e-5 for --tune full)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj",
                        help="comma-separated module name substrings to LoRA")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine",
                        help="cosine: linear warmup then cosine decay down to --min-lr-ratio; "
                             "none: keep --lr flat for the whole run")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="fraction of all steps spent on linear warmup (cosine schedule)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="final lr as a fraction of the base lr (cosine floor)")
    parser.add_argument("--loss-chunk", type=int, default=256,
                        help="positions per LM-head chunk when computing the loss; the chunk's logits "
                             "are recomputed in backward, so this bounds the vocab-sized activation "
                             "memory (<=0 = one shot through the head, the memory-hungry old path)")
    parser.add_argument("--loss-ema", type=float, default=0.02,
                        help="EMA weight for train/loss_ema (~35-step half-life); 0 disables smoothing")
    parser.add_argument("--lang-filter", default=None,
                        help="only train on this lang (zh/en); None = all")
    parser.add_argument("--limit", type=int, default=0, help="limit examples (smoke test)")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="apply random waveform augmentation inside the training dataset")
    parser.add_argument("--augment-mel", action=argparse.BooleanOptionalAction, default=True,
                        help="apply SpecAugment masks after the encoder frontend")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="fixed instruction prepended to every stage-2 row (audio-only data)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR[args.tune]

    torch.manual_seed(args.seed)
    ddp_utils.setup()
    device = ddp_utils.device()
    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed. Run: python -m pip install -r requirements.txt")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # ---- load frozen frontend ----
    encoder = build_frozen_encoder(
        args.encoder_type,
        checkpoint=str(args.encoder_checkpoint) if args.encoder_type == "conformer" else None,
        model_id=(args.sensevoice_model if args.encoder_type == "sensevoice" else args.paraformer_model),
        device=device,
    )
    acoustic_dim = encoder.output_dim

    # ---- load Qwen3 + prepare fine-tuning (lora or full) ----
    lm, tokenizer = load_qwen3(
        args.qwen3_model, device, dtype=None if args.llm_dtype == "auto" else args.llm_dtype
    )
    for p in lm.parameters():
        p.requires_grad_(False)

    proj_ckpt = torch.load(args.projector_checkpoint, map_location=device, weights_only=False)
    llm_dim = int(proj_ckpt.get("llm_hidden_size", lm.config.hidden_size))
    if llm_dim != int(lm.config.hidden_size):
        raise SystemExit(
            f"projector was trained against a {llm_dim}-dim backbone but "
            f"{args.qwen3_model} is {lm.config.hidden_size}-dim; retrain the projector "
            "(trainer/train_speech_projector.py) against this Qwen3 checkpoint first"
        )
    projector = SpeechProjector(acoustic_dim, llm_dim).to(device)
    projector.load_state_dict(proj_ckpt["projector"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(args.tune_projector)
    # DDP refuses to wrap a module whose parameters are all frozen, and a frozen
    # projector has no gradient to synchronise (same reason the encoder is never
    # wrapped), so it stays a plain module.
    if args.tune_projector:
        projector = ddp_utils.wrap(projector)

    if args.tune == "lora":
        lm = add_lora(
            lm,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=[m for m in args.lora_targets.split(",") if m],
        )
    else:  # full: unfreeze every LLM parameter (frontend stays frozen)
        for p in lm.parameters():
            p.requires_grad_(True)
        print("full fine-tune: all Qwen3 parameters trainable")

    base_lm = ddp_utils.unwrap(lm)
    lm = ddp_utils.wrap(lm)
    projector_module = ddp_utils.unwrap(projector)
    trainable_params = [p for p in base_lm.parameters() if p.requires_grad]
    optimizer_groups = [{"params": trainable_params, "lr": args.lr}]
    projector_params = [p for p in projector_module.parameters() if p.requires_grad]
    if projector_params:
        optimizer_groups.append({"params": projector_params, "lr": args.projector_lr or args.lr})
    optimizer = torch.optim.AdamW(optimizer_groups)

    # ---- data ---- (train/dev are separate manifests produced by the splitter)
    args.output.mkdir(parents=True, exist_ok=True)

    def resolve(manifest, training: bool):
        config = SpeechAugmentConfig(enabled=args.augment and training)
        return SpeechInstructionDataset(
            manifest,
            system_prompt=args.system_prompt,
            augment=config,
            lang_filter=args.lang_filter,
            limit=args.limit,
        )

    if args.data.is_dir():
        train_manifest = args.data / "train.jsonl"
        # "dev" is the trainer's name; accept "val" as produced by the splitter.
        dev_manifest = next(
            (args.data / name for name in ("dev.jsonl", "val.jsonl")
             if (args.data / name).exists()),
            None,
        )
    else:
        train_manifest = args.data
        dev_manifest = next(
            (args.data.parent / name for name in ("dev.jsonl", "val.jsonl")
             if (args.data.parent / name).exists()),
            None,
        )
    if not train_manifest.exists():
        raise FileNotFoundError(f"train manifest not found: {train_manifest}")
    if dev_manifest is None:
        raise FileNotFoundError(
            f"no dev/val manifest next to {train_manifest}; run "
            "scripts/synthesize_stage2_tts_and_split.py to produce the train/val/test splits"
        )

    train_set = resolve(train_manifest, training=True)
    dev_set = resolve(dev_manifest, training=False)

    train_sampler = ddp_utils.make_sampler(train_set, shuffle=True)
    dev_sampler = ddp_utils.make_sampler(dev_set, shuffle=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=dev_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # ---- lr schedule ---- one step per optimizer step, identical on every rank
    # (DistributedSampler pads, so all ranks run the same number of batches).
    scheduler = None
    total_steps = 0
    warmup_steps = 0
    if args.lr_schedule == "cosine":
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = max(1, int(args.warmup_ratio * total_steps))
        floor = args.min_lr_ratio

        def lr_scale(step: int) -> float:
            """Linear warmup, then cosine decay to ``floor`` of each group's base lr."""
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    # save config + initial model/adapter reference (rank 0 only)
    if ddp_utils.is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        base_name = "qwen3_base_lora" if args.tune == "lora" else "qwen3_base"
        base_lm.save_pretrained(args.output / base_name)
        tokenizer.save_pretrained(args.output / base_name)
        (args.output / "config.json").write_text(
            json.dumps(vars(args) | {"device": str(device), "llm_hidden_size": llm_dim},
                       default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv
            csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writeheader()

    if ddp_utils.is_main():
        print(f"device: {device}")
        print(f"llm_hidden_size: {llm_dim}")
        print(f"tune_mode: {args.tune}  llm_lr: {args.lr:g}")
        print(f"tune_projector: {args.tune_projector}")
        print(f"train_set_rows: {len(train_set)}  dev_set_rows: {len(dev_set)}")
        print(f"trainable: {sum(p.numel() for p in trainable_params):,} "
              f"({sum(p.numel() for p in trainable_params) * 100.0 / max(sum(p.numel() for p in base_lm.parameters()), 1):.2f}% of LLM)")
        if projector_params:
            print(f"projector_trainable: {sum(p.numel() for p in projector_params):,} "
                  f"(lr={args.projector_lr or args.lr:g})")
        if scheduler is None:
            print(f"lr_schedule: none (flat lr {args.lr:g})")
        else:
            print(f"lr_schedule: cosine  warmup_steps: {warmup_steps}  total_steps: {total_steps}  "
                  f"min_lr_ratio: {args.min_lr_ratio:g} (lr floor {args.lr * args.min_lr_ratio:g})")
        print(f"loss_ema: {args.loss_ema:g}")

    loss_ema = None
    for epoch in range(1, args.epochs + 1):
        train_loss, loss_ema = run_epoch(
            args, encoder, projector, lm, tokenizer, train_loader, device, optimizer,
            sampler=train_sampler, epoch=epoch, scheduler=scheduler, loss_ema=loss_ema,
        )
        with torch.no_grad():
            dev_loss, _ = run_epoch(
                args, encoder, projector, lm, tokenizer, dev_loader, device,
            )
        if ddp_utils.is_main():
            if args.tune == "full":
                # full: save whole model + tokenizer (safetensors / bin)
                base_lm.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
                tokenizer.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
                ckpt_tag = f"model_epoch_{epoch:03d}"
            else:
                base_lm.save_pretrained(args.output / f"lora_epoch_{epoch:03d}")
                ckpt_tag = f"lora_epoch_{epoch:03d}"
            if args.tune_projector:
                projector_path = args.output / f"projector_epoch_{epoch:03d}.pt"
                torch.save({
                    "projector": projector_module.state_dict(),
                    "acoustic_dim": acoustic_dim,
                    "llm_hidden_size": llm_dim,
                    "source_checkpoint": str(args.projector_checkpoint),
                }, projector_path)
            with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
                import csv
                csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writerow(
                    {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "dev_loss": f"{dev_loss:.6f}"})
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} -> {ckpt_tag}")
        if args.wandb and ddp_utils.is_main():
            wandb.log({"train/loss": train_loss, "dev/loss": dev_loss, "epoch": epoch})
    if args.wandb and ddp_utils.is_main():
        wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
