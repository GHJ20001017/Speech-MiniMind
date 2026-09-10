"""Train the 4-layer Tiny Conformer on AISHELL-1 with character CTC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import ddp_utils  # noqa: E402
from model.ctc_model import TinyConformerCTC  # noqa: E402
from scripts.analyze_audio import log_mel, read_wav  # noqa: E402


class AishellDataset(Dataset):
    def __init__(self, manifest: Path, vocab: dict[str, int]) -> None:
        with manifest.open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio, rate = read_wav(Path(row["path"]))
        features, _, _, _ = log_mel(audio, rate, 25, 10, 80)
        target = torch.tensor([self.vocab[c] for c in row["text"] if c in self.vocab], dtype=torch.long)
        return torch.from_numpy(features.astype(np.float32)), target


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, targets = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    target_lengths = torch.tensor([item.numel() for item in targets], dtype=torch.long)
    padded_features = pad_sequence(features, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    return padded_features, lengths, padded_targets, target_lengths


def evaluate(model, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    base = ddp_utils.unwrap(model)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, lengths, targets, target_lengths in loader:
            features, targets = features.to(device), targets.to(device)
            frame_steps = torch.arange(features.size(1), device=device).unsqueeze(0)
            padding_mask = frame_steps >= lengths.to(device).unsqueeze(1)
            logits = model(features, padding_mask)
            input_lengths = base.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
            loss = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
            total_loss += loss.detach().item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/02_acoustic_encoder"))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    ddp_utils.setup()
    vocab = {line.strip(): index for index, line in enumerate((args.data / "vocab.txt").read_text(encoding="utf-8").splitlines()) if line.strip()}
    train_set = AishellDataset(args.data / "train.csv", vocab)
    dev_set = AishellDataset(args.data / "dev.csv", vocab)
    train_sampler = ddp_utils.make_sampler(train_set, shuffle=True)
    dev_sampler = ddp_utils.make_sampler(dev_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, sampler=dev_sampler, collate_fn=collate)
    device = ddp_utils.device()
    model = ddp_utils.wrap(TinyConformerCTC(len(vocab)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed. Run: python -m pip install -r requirements.txt")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    metrics_path = args.output / "metrics.csv"
    if ddp_utils.is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(json.dumps({
            "data": str(args.data), "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device), "vocab_size": len(vocab),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with metrics_path.open("w", newline="") as metrics_file:
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=["epoch", "train_ctc_loss", "dev_ctc_loss", "learning_rate", "seconds"])
            metrics_writer.writeheader()
        print(f"device: {device}")
        print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        ddp_utils.set_epoch(train_sampler, epoch)
        base = ddp_utils.unwrap(model)
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch:02d}/{args.epochs}", unit="batch", disable=not ddp_utils.is_main())
        batch_count = 0
        for features, lengths, targets, target_lengths in progress:
            features, targets = features.to(device), targets.to(device)
            frame_steps = torch.arange(features.size(1), device=device).unsqueeze(0)
            padding_mask = frame_steps >= lengths.to(device).unsqueeze(1)
            logits = model(features, padding_mask)
            input_lengths = base.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
            loss = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = loss.detach().item()
            total_loss += loss_value
            batch_count += 1
            progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total_loss / batch_count:.4f}")
            if args.wandb and ddp_utils.is_main():
                wandb.log({"train/ctc_loss_step": loss_value, "step": epoch * len(train_loader) + batch_count})
        train_loss = ddp_utils.all_reduce_mean(total_loss / max(len(train_loader), 1))
        dev_loss = ddp_utils.all_reduce_mean(evaluate(model, dev_loader, device, loss_fn))
        seconds = time.perf_counter() - epoch_start
        if ddp_utils.is_main():
            with metrics_path.open("a", newline="") as metrics_file:
                csv.DictWriter(metrics_file, fieldnames=["epoch", "train_ctc_loss", "dev_ctc_loss", "learning_rate", "seconds"]).writerow({
                    "epoch": epoch, "train_ctc_loss": f"{train_loss:.6f}", "dev_ctc_loss": f"{dev_loss:.6f}",
                    "learning_rate": f"{args.lr:.8g}", "seconds": f"{seconds:.2f}",
                })
            if args.wandb:
                wandb.log({
                    "train/ctc_loss": train_loss,
                    "dev/ctc_loss": dev_loss,
                    "train/learning_rate": args.lr,
                    "epoch": epoch,
                    "epoch_seconds": seconds,
                })
            checkpoint = args.output / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({"model": base.state_dict(), "vocab": vocab, "epoch": epoch, "train_ctc_loss": train_loss, "dev_ctc_loss": dev_loss}, checkpoint)
            print(f"epoch={epoch:02d} train_ctc_loss={train_loss:.4f} dev_ctc_loss={dev_loss:.4f} seconds={seconds:.1f}")
    if ddp_utils.is_main():
        final_checkpoint = args.output / "tiny_conformer_ctc.pt"
        torch.save({"model": ddp_utils.unwrap(model).state_dict(), "vocab": vocab, "epoch": args.epochs}, final_checkpoint)
        metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        epochs = [int(row["epoch"]) for row in metrics]
        train_losses = [float(row["train_ctc_loss"]) for row in metrics]
        dev_losses = [float(row["dev_ctc_loss"]) for row in metrics]
        figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
        axis.plot(epochs, train_losses, marker="o", markersize=4, linewidth=2, label="train")
        axis.plot(epochs, dev_losses, marker="o", markersize=4, linewidth=2, label="dev")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("CTC loss")
        axis.set_title("Tiny Conformer CTC loss")
        axis.set_xticks(epochs)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(args.output / "loss_curve.png")
        plt.close(figure)
        print(f"metrics_saved: {metrics_path}")
        print(f"loss_curve_saved: {args.output / 'loss_curve.png'}")
        print(f"checkpoint_saved: {final_checkpoint}")
        if args.wandb:
            wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
