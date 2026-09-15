"""B1/B2: speech-to-speech fine-tuning of the audio-native LLM.

Continues from the B0 audio-LM checkpoint.  Each sample pairs a *prompt*
utterance with an *answer* utterance; the model reads the prompt codes and
generates the answer codes, and the loss covers only the answer span::

    [BOS] <|audio_bos|> prompt codes <|audio_eos|>
          <|audio_bos|> answer codes <|audio_eos|> [EOS]

Use ``--init-from`` to start from the B0 output directory (or any directory that
already contains the audio-extended vocabulary).  When starting from a plain
MiniMind checkpoint, the audio block is added and randomly initialised, which is
only recommended for a quick smoke test.

Usage::

    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
        trainer/train_speech_to_speech.py \
        --data data/route_b/s2s \
        --init-from outputs/05_route_b_audio_lm/model_epoch_003 \
        --output outputs/06_route_b_s2s --epochs 3 --batch-size 4 \
        --tune full --wandb --wandb-name route_b_s2s
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

from dataset.audio_token_dataset import SpeechToSpeechDataset  # noqa: E402
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
    axis.set(xlabel="Epoch", ylabel="Answer-token CE loss", title="Route B S2S loss")
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
            tokenizer, spec, samples,
            max_length=args.max_length,
            max_answer_frames=args.max_answer_frames,
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

    mean_loss = total_loss / max(total_tokens, 1)
    return ddp_utils.all_reduce_mean(mean_loss)


@torch.no_grad()
def token_accuracy(model, tokenizer, spec, loader, device, args) -> float:
    """Teacher-forced next-token accuracy over the supervised answer span."""
    model.eval()
    correct = 0.0
    total = 0.0
    for samples in loader:
        input_ids, labels, attention_mask = build_audio_batch(
            tokenizer, spec, samples,
            max_length=args.max_length,
            max_answer_frames=args.max_answer_frames,
        )
        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attention_mask.to(device)).logits
        predictions = logits[:, :-1].argmax(dim=-1)
        targets = labels[:, 1:].to(device)
        mask = targets != -100
        correct += (predictions[mask] == targets[mask]).sum().item()
        total += mask.sum().item()
    return ddp_utils.all_reduce_mean(correct / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="dir with {train,dev}.jsonl pointing at .npy code shards")
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="B0 checkpoint dir (with audio-extended vocab); defaults to the raw MiniMind")
    parser.add_argument("--minimind-model", type=Path, default=None,
                        help="raw MiniMind dir, used when --init-from is omitted")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("outputs/06_route_b_s2s"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=0,
                        help="hard cap on the token sequence length; 0 = auto from the frame caps "
                             "((max_prompt_frames+max_answer_frames)*num_codebooks + 16)")
    parser.add_argument("--max-answer-frames", type=int, default=256,
                        help="cap supervised answer frames per sample")
    parser.add_argument("--max-prompt-frames", type=int, default=256)
    parser.add_argument("--code-dropout", type=float, default=0.0,
                        help="randomly corrupt this fraction of prompt frames (robustness)")
    parser.add_argument("--tune", choices=("full", "embed"), default="full")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lang-filter", default=None, choices=(None, "zh", "en", "mixed"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.max_length <= 0:
        # Both the prompt and the answer occupy (frames * num_codebooks) tokens.
        args.max_length = (args.max_prompt_frames + args.max_answer_frames) * args.num_codebooks + 16

    torch.manual_seed(args.seed)
    ddp_utils.setup()
    device = ddp_utils.device()

    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed. Run: python -m pip install -r requirements.txt")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    source = args.init_from or args.minimind_model
    if source is None:
        raise SystemExit("pass --init-from (B0 checkpoint) or --minimind-model (raw MiniMind)")

    model, tokenizer = load_minimind(source, device)
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    def resolve(manifest: Path) -> SpeechToSpeechDataset:
        return SpeechToSpeechDataset(
            manifest,
            max_answer_frames=args.max_answer_frames,
            max_prompt_frames=args.max_prompt_frames,
            code_dropout_prob=args.code_dropout,
            codebook_size=args.codebook_size,
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
            csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss", "dev_token_acc"]).writeheader()
        print(f"device: {device}")
        print(f"train_rows: {len(train_set)}  dev_rows: {len(dev_set)}")
        print(f"trainable: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(args, model, tokenizer, spec, train_loader, device,
                               optimizer, sampler=train_sampler, epoch=epoch)
        with torch.no_grad():
            dev_loss = run_epoch(args, model, tokenizer, spec, dev_loader, device)
            dev_acc = token_accuracy(model, tokenizer, spec, dev_loader, device, args)
        if ddp_utils.is_main():
            base_model.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
            tokenizer.save_pretrained(args.output / f"model_epoch_{epoch:03d}")
            with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=["epoch", "train_loss", "dev_loss", "dev_token_acc"],
                ).writerow({
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "dev_loss": f"{dev_loss:.6f}",
                    "dev_token_acc": f"{dev_acc:.6f}",
                })
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
                  f"dev_token_acc={dev_acc:.4f}")
            save_loss_curve(args.output / "metrics.csv", args.output / "loss_curve.png")
        if args.wandb and ddp_utils.is_main():
            wandb.log({"train/loss": train_loss, "dev/loss": dev_loss,
                       "dev/token_acc": dev_acc, "epoch": epoch})

    if args.wandb and ddp_utils.is_main():
        wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
