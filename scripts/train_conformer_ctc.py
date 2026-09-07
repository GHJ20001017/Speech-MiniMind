"""Train the 4-layer Tiny Conformer on AISHELL-1 with character CTC."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    vocab = {line.strip(): index for index, line in enumerate((args.data / "vocab.txt").read_text(encoding="utf-8").splitlines()) if line.strip()}
    train_loader = DataLoader(AishellDataset(args.data / "train.csv", vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyConformerCTC(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    print(f"device: {device}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, lengths, targets, target_lengths in train_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            input_lengths = torch.div(lengths, 4, rounding_mode="floor").clamp_min(1)
            loss = loss_fn(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss)
        print(f"epoch={epoch:02d} train_ctc_loss={total_loss / max(len(train_loader), 1):.4f}")
    checkpoint = args.data / "tiny_conformer_ctc.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab}, checkpoint)
    print(f"checkpoint_saved: {checkpoint}")


if __name__ == "__main__":
    main()
