"""Train the Chapter 02 log-Mel audio classifier."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise SystemExit("PyTorch is required. Install dependencies with: python -m pip install -r requirements.txt") from error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_audio import log_mel, read_wav  # noqa: E402
from model.audio_classifier import AudioClassifier  # noqa: E402


class AudioDataset(Dataset):
    def __init__(self, manifest: Path, n_mels: int, frame_ms: float, hop_ms: float) -> None:
        with manifest.open() as handle:
            self.rows = list(csv.DictReader(handle))
        self.n_mels, self.frame_ms, self.hop_ms = n_mels, frame_ms, hop_ms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio, sample_rate = read_wav(Path(row["path"]))
        features, _, _, _ = log_mel(audio, sample_rate, self.frame_ms, self.hop_ms, self.n_mels)
        return torch.from_numpy(features.astype(np.float32)), torch.tensor(int(row["label"]), dtype=torch.long)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/audio_classification"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = AudioDataset(args.data / "train.csv", 80, 25, 10)
    valid_set = AudioDataset(args.data / "valid.csv", 80, 25, 10)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate)
    model = AudioClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
        accuracy = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:02d} loss={running_loss / len(train_loader):.4f} valid_accuracy={accuracy:.3f}")
    args.data.joinpath("audio_classifier.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.data / "audio_classifier.pt")
    print(f"checkpoint_saved: {args.data / 'audio_classifier.pt'}")


if __name__ == "__main__":
    main()
