"""Instruction-tune a Speech→MiniMind LLM on speech instruction data.

Pipeline:

    audio ──> TinyConformer.encoder (frozen) ──> SpeechProjector
                ──> speech prefix embeddings
                ──concat──> MiniMind (LoRA or full fine-tuned) ──> answer text

The acoustic encoder is always frozen. The SpeechProjector is frozen by default
and can optionally be fine-tuned with ``--tune-projector`` when the target
MiniMind and speech frontend need to adapt together.

Data: any JSONL with ``audio, instruction, answer`` (e.g. the merged
``data/stage2_mixed/{train,dev}.jsonl`` produced by
``merge_speech_instruction_datasets.py``). The ``lang``/``task`` fields are
ignored by training but may optionally filter rows.

Loss is computed only over the ``answer`` portion (prompt tokens are masked).

Requires: peft (``pip install peft``), plus the same torch/transformers as the
rest of the project. Mimics ``forward_inputs_embeds`` for the LLM forward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.frozen_encoder import build_frozen_encoder  # noqa: E402
from model import ddp_utils  # noqa: E402
from model.minimind_adapter import forward_inputs_embeds, load_minimind  # noqa: E402
from model.speech_projector import SpeechProjector  # noqa: E402
from scripts.analyze_audio import read_wav  # noqa: E402

SAMPLE_RATE = 16000

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None


class SpeechInstructionDataset(Dataset):
    """Read merged JSONL (``audio,instruction,answer``) into speech features."""

    def __init__(self, manifest: Path, lang_filter: str | None = None) -> None:
        rows: list[dict[str, str]] = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if lang_filter and record.get("lang") != lang_filter:
                    continue
                audio = str(record.get("audio", "")).strip()
                instruction = str(record.get("instruction", "")).strip()
                answer = str(record.get("answer", "")).strip()
                if not audio or not answer:
                    continue
                rows.append({"audio": audio, "instruction": instruction, "answer": answer})
        self.rows = rows
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.rows[index]
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = self.manifest.parent / audio_path
        audio_array, rate = read_wav(audio_path)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, got {rate} (path={audio_path})")
        return torch.from_numpy(audio_array.astype(np.float32)), rate, row["instruction"], row["answer"]


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

    Layout per sample: [speech tokens][instruction tokens][answer tokens]
    Labels mark only the answer portion as trainable.
    """
    seqs_emb: list[torch.Tensor] = []
    seqs_mask: list[torch.Tensor] = []
    seqs_label: list[torch.Tensor] = []
    for speech, speech_len, instruction, answer in zip(
        projected, projected_lengths.tolist(), instructions, answers
    ):
        speech = speech[: min(speech_len, max_speech_tokens, max_length - 2)]
        prompt_ids = tokenizer(
            instruction, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0].to(device)
        target_ids = tokenizer(
            answer + (tokenizer.eos_token or ""), add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(device)

        all_ids = torch.cat([prompt_ids, target_ids])[: max_length - speech.size(0)]
        prompt_count = min(prompt_ids.numel(), all_ids.numel())

        # `model` may be a PeftModel (optionally DDP-wrapped); unwrap to reach
        # the base LM's embeddings.
        base_model = ddp_utils.unwrap(model)
        base_model = base_model.get_base_model() if hasattr(base_model, "get_base_model") else base_model
        text_embeds = base_model.model.embed_tokens(all_ids).unsqueeze(0)  # [1, T, H]

        emb = torch.cat([speech.unsqueeze(0), text_embeds], dim=1)  # [1, S+T, H]
        seq_len = emb.size(1)
        pad_len = max_length - seq_len
        if pad_len > 0:
            emb = torch.cat([emb, torch.zeros(1, pad_len, emb.size(2), device=device)], dim=1)
        mask = torch.zeros(max_length, device=device, dtype=torch.long)
        mask[:seq_len] = 1
        label = torch.full((max_length,), -100, device=device, dtype=torch.long)
        label[speech.size(0) + prompt_count : seq_len] = all_ids[prompt_count:]

        seqs_emb.append(emb)
        seqs_mask.append(mask)
        seqs_label.append(label)

    return torch.cat(seqs_emb, dim=0), torch.stack(seqs_mask), torch.stack(seqs_label)


def run_epoch(args, encoder, projector, lm, tokenizer, loader, device, optimizer=None, sampler=None, epoch=0):
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
            acoustic, acoustic_lengths = encoder.encode(waveforms.to(device), lengths.to(device), SAMPLE_RATE)
        projected = projector(acoustic)
        projected_lengths = projector_module.output_lengths(acoustic_lengths).clamp_max(projected.size(1))

        inputs_embeds, attention_mask, labels = make_sft_batch(
            lm, tokenizer, projected, projected_lengths, instructions, answers,
            device, args.max_length, args.max_speech_tokens,
        )
        with torch.set_grad_enabled(training):
            logits = forward_inputs_embeds(lm, inputs_embeds, attention_mask)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(lm.parameters()) + list(projector.parameters()) if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            if args.wandb and ddp_utils.is_main():
                wandb.log({"train/loss_step": loss.detach().item()})
        total_loss += loss.detach().item()
        steps += 1
    return ddp_utils.all_reduce_mean(total_loss / max(steps, 1))


def add_lora(
    model,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str] | None = None,
):
    """Apply LoRA low-rank adapters to MiniMind attention projections in place.

    Uses peft. Raises if peft is missing.
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
                        help="dir or single JSONL with {train,dev}.jsonl (audio/instruction/answer)")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--encoder-type", choices=("conformer", "paraformer"), default="conformer",
                        help="frozen acoustic encoder backend")
    parser.add_argument("--paraformer-model", default=None,
                        help="FunASR model id/dir for --encoder-type paraformer")
    parser.add_argument("--projector-checkpoint", type=Path, required=True,
                        help="trained SpeechProjector ckpt (outputs/03_speech_minimind/projector_epoch_XXX.pt)")
    parser.add_argument("--tune-projector", action=argparse.BooleanOptionalAction, default=False,
                        help="also fine-tune SpeechProjector; default keeps the frontend frozen")
    parser.add_argument("--projector-lr", type=float, default=None,
                        help="Projector learning rate when --tune-projector (default: --lr)")
    parser.add_argument("--minimind-model", type=Path, required=True,
                        help="local Transformers-format MiniMind dir")
    parser.add_argument("--tune", choices=("lora", "full"), default="lora",
                        help="lora: LoRA adapters only (~0.5%% trainable, needs peft); "
                             "full: full-parameter LLM fine-tune (all LLM weights trainable)")
    parser.add_argument("--output", type=Path, default=Path("outputs/04_speech_minimind_sft"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj",
                        help="comma-separated module name substrings to LoRA")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lang-filter", default=None,
                        help="only train on this lang (zh/en); None = all")
    parser.add_argument("--limit", type=int, default=0, help="limit examples (smoke test)")
    parser.add_argument("--dev-file", type=Path, default=None,
                        help="dev JSONL when --data is a single train JSONL")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

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
        model_id=args.paraformer_model,
        device=device,
    )
    acoustic_dim = encoder.output_dim

    proj_ckpt = torch.load(args.projector_checkpoint, map_location=device, weights_only=False)
    llm_dim = int(proj_ckpt.get("llm_hidden_size", 768))
    projector = SpeechProjector(acoustic_dim, llm_dim).to(device)
    projector.load_state_dict(proj_ckpt["projector"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(args.tune_projector)
    projector = ddp_utils.wrap(projector)

    # ---- load LLM + prepare fine-tuning (lora or full) ----
    lm, tokenizer = load_minimind(args.minimind_model, device)
    for p in lm.parameters():
        p.requires_grad_(False)

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
        print("full fine-tune: all MiniMind parameters trainable")

    base_lm = ddp_utils.unwrap(lm)
    lm = ddp_utils.wrap(lm)
    projector_module = ddp_utils.unwrap(projector)
    trainable_params = [p for p in base_lm.parameters() if p.requires_grad]
    optimizer_groups = [{"params": trainable_params, "lr": args.lr}]
    projector_params = [p for p in projector_module.parameters() if p.requires_grad]
    if projector_params:
        optimizer_groups.append({"params": projector_params, "lr": args.projector_lr or args.lr})
    optimizer = torch.optim.AdamW(optimizer_groups)

    # ---- data ----
    args.output.mkdir(parents=True, exist_ok=True)

    def resolve(manifest):
        return SpeechInstructionDataset(manifest, args.lang_filter)

    if args.data.is_dir():
        train_manifest, dev_manifest = args.data / "train.jsonl", args.data / "dev.jsonl"
        if not train_manifest.exists() or not dev_manifest.exists():
            raise FileNotFoundError(f"need {args.data}/{{train,dev}}.jsonl")
    else:
        if not args.dev_file:
            raise SystemExit("For single-file data, pass --dev-file <path>")
        train_manifest, dev_manifest = args.data, args.dev_file

    train_set = resolve(train_manifest)
    dev_set = resolve(dev_manifest)
    if args.limit:
        train_set.rows = train_set.rows[: args.limit]
        dev_set.rows = dev_set.rows[: min(args.limit, len(dev_set))]

    train_sampler = ddp_utils.make_sampler(train_set, shuffle=True)
    dev_sampler = ddp_utils.make_sampler(dev_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, sampler=dev_sampler, collate_fn=collate)

    # save config + initial model/adapter reference (rank 0 only)
    if ddp_utils.is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        base_name = "minimind_base_lora" if args.tune == "lora" else "minimind_base"
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
        print(f"tune_mode: {args.tune}")
        print(f"tune_projector: {args.tune_projector}")
        print(f"train_set_rows: {len(train_set)}  dev_set_rows: {len(dev_set)}")
        print(f"trainable: {sum(p.numel() for p in trainable_params):,} "
              f"({sum(p.numel() for p in trainable_params) * 100.0 / max(sum(p.numel() for p in base_lm.parameters()), 1):.2f}% of LLM)")
        if projector_params:
            print(f"projector_trainable: {sum(p.numel() for p in projector_params):,} "
                  f"(lr={args.projector_lr or args.lr:g})")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(args, encoder, projector, lm, tokenizer,
                               train_loader, device, optimizer, sampler=train_sampler, epoch=epoch)
        with torch.no_grad():
            dev_loss = run_epoch(args, encoder, projector, lm, tokenizer,
                                 dev_loader, device)
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