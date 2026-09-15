"""B0: pre-train an audio-native LLM on discrete codebook tokens.

This is the Route-B first stage: before any speech-to-speech pairing exists, the
LM learns the *distribution* of the codec's tokens, exactly as a text LM learns
a language.  Every sample is one utterance's code sequence wrapped as

    [BOS] <|audio_start|> codes... <|audio_end|> [EOS]

and the whole code span is supervised.

Model: MiniMind decoder + an audio token block appended after the text vocab
(see ``model/audio_lm.py``).  Because the audio rows are freshly initialised,
``--tune full`` is the default - LoRA cannot reach the new embedding rows.

Usage::

    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
        trainer/train_audio_lm_pretrain.py \
        --data data/route_b/audio_lm_codes \
        --minimind-model /gpu3/guhj/models/minimind-3 \
        --output outputs/05_route_b_audio_lm --epochs 3 --batch-size 8 \
        --tune full --wandb --wandb-name route_b_b0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.audio_token_dataset import AudioLMDataset  # noqa: E402
from model import ddp_utils  # noqa: E402
from model.audio_lm import (  # noqa: E402
    AudioSample,
    build_audio_batch,
    build_vocab_spec,
    extend_model_vocab,
    register_audio_special_tokens,
)
from model.minimind_adapter import load_minimind  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None


def collate(samples: list[AudioSample]) -> list[AudioSample]:
    return samples


def build_optimizer(model, lr: float, weight_decay: float = 0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def save_loss_curve(metrics: Path, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting optional
        return
    with metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    axis.plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", label="Train")
    axis.plot(epochs, [float(r["dev_loss"]) for r in rows], marker="o", label="Dev")
    axis.set(xlabel="Epoch", ylabel="Token CE loss", title="Route B B0 audio-LM loss")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def run_epoch(args, model, tokenizer, spec, loader, device, optimizer=None, sampler=None, epoch=0):
    training = optimizer is not None
    if sampler is not None:
        ddp_utils.set_epoch(sampler, epoch)
    model.train(training)
    total_loss = 0.0
    total_tokens = 0.0
    steps = 0
    for samples in tqdm(loader, desc="train" if training else "dev", unit="batch",
                        disable=not ddp_utils.is_main()):
        input_ids, labels, attention_mask = build_audio_batch(
            tokenizer, spec, samples, max_length=args.max_length,
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        with torch.set_grad_enabled(training):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()
            if args.wandb and ddp_utils.is_main():
                wandb.log({"train/loss_step": loss.detach().item()})

        supervised = (labels[:, 1:] != -100).sum().item()
        total_loss += loss.detach().item() * supervised
        total_tokens += supervised
        steps += 1

    # Weight by supervised tokens, then average across ranks.
    mean_loss = total_loss / max(total_tokens, 1)
    return ddp_utils.all_reduce_mean(mean_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="dir with {train,dev}.jsonl whose rows point at .npy code shards")
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--minimind-model", type=Path, required=True)
    parser.add_argument("--num-codebooks", type=int, default=8,
                        help="codebooks per frame (Mimi=8; single-codebook codecs use 1)")
    parser.add_argument("--codebook-size", type=int, default=2048,
                        help="codes per codebook (Mimi=2048)")
    parser.add_argument("--output", type=Path, default=Path("outputs/05_route_b_audio_lm"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=0,
                        help="hard cap on the token sequence length; 0 = auto from the frame caps "
                             "(max_frames*num_codebooks + 16)")
    parser.add_argument("--max-frames", type=int, default=128,
                        help="longest utterance kept per sample, in code frames")
    parser.add_argument("--tune", choices=("full", "embed"), default="full",
                        help="full: train the whole LM; embed: only the audio embeddings/head")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="limit rows (smoke test)")
    parser.add_argument("--lang-filter", default=None, choices=(None, "zh", "en"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.max_length <= 0:
        # One frame = num_codebooks tokens; leave headroom for the wrapper tokens
        # so the sequence cap never silently truncates supervised tokens.
        args.max_length = args.max_frames * args.num_codebooks + 16

    torch.manual_seed(args.seed)
    ddp_utils.setup()
    device = ddp_utils.device()

    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed. Run: python -m pip install -r requirements.txt")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    model, tokenizer = load_minimind(args.minimind_model, device)
    register_audio_special_tokens(tokenizer)
    spec = build_vocab_spec(tokenizer, args.codebook_size, args.num_codebooks)
    spec = extend_model_vocab(model, tokenizer, spec)
    print(f"vocab: text={spec.text_vocab_size} audio={spec.audio_vocab_size} "
          f"total={spec.total_vocab_size}")

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if args.tune == "embed":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("model.embed_tokens") or name.startswith("lm_head")
            )

    base_model = ddp_utils.unwrap(model)
    model = ddp_utils.wrap(model)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    def resolve(manifest: Path) -> AudioLMDataset:
        return AudioLMDataset(
            manifest,
            max_frames=args.max_frames,
            lang_filter=args.lang_filter,
            limit=args.limit,
        )

    if args.data.is_dir():
        train_manifest, dev_manifest = args.data / "train.jsonl", args.data / "dev.jsonl"
    else:
        if not args.dev_file:
            raise SystemExit("For a single-file --data, pass --dev-file")
        train_manifest, dev_manifest = args.data, args.dev_file

    train_set = resolve(train_manifest)
    dev_set = resolve(dev_manifest)

    train_sampler = ddp_utils.make_sampler(train_set, shuffle=True)
    dev_sampler = ddp_utils.make_sampler(dev_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda",
                              persistent_workers=args.num_workers > 0)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                            sampler=dev_sampler, collate_fn=collate,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda",
                            persistent_workers=args.num_workers > 0)

    if ddp_utils.is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(args.output / "minimind_audio_base")
        tokenizer.save_pretrained(args.output / "minimind_audio_base")
        (args.output / "config.json").write_text(
            json.dumps(vars(args) | {
                "device": str(device),
                "text_vocab_size": spec.text_vocab_size,
                "audio_offset": spec.audio_offset,
                "audio_vocab_size": spec.audio_vocab_size,
                "num_codebooks": spec.num_codebooks,
                "codebook_size": spec.codebook_size,
            }, default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writeheader()
        print(f"device: {device}")
        print(f"train_rows: {len(train_set)}  dev_rows: {len(dev_set)}")
        print(f"trainable: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(args, model, tokenizer, spec, train_loader, device,
                               optimizer, sampler=train_sampler, epoch=epoch)
        with torch.no_grad():
            dev_loss = run_epoch(args, model, tokenizer, spec, dev_loader, device)
        if ddp_utils.is_main():
            base_model.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
            tokenizer.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
            with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writerow(
                    {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "dev_loss": f"{dev_loss:.6f}"})
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")
            save_loss_curve(args.output / "metrics.csv", args.output / "loss_curve.png")
        if args.wandb and ddp_utils.is_main():
            wandb.log({"train/loss": train_loss, "dev/loss": dev_loss, "epoch": epoch})

    if args.wandb and ddp_utils.is_main():
        wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
