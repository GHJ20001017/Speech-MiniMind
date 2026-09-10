"""Evaluate a Tiny Conformer CTC checkpoint with greedy decoding and CER."""

from __future__ import annotations

import argparse
import csv
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
from model.ctc_model import TinyConformerCTC  # noqa: E402
from model.ctc_streaming import TinyStreamingConformerCTC  # noqa: E402
from scripts.analyze_audio import log_mel, read_wav  # noqa: E402


class AishellEvalDataset(Dataset):
    def __init__(self, manifest: Path) -> None:
        with manifest.open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[index]
        audio, rate = read_wav(Path(row["path"]))
        features, _, _, _ = log_mel(audio, rate, 25, 10, 80)
        return torch.from_numpy(features.astype(np.float32)), row["text"], row["path"]


def collate(batch: list[tuple[torch.Tensor, str, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    features, texts, paths = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    return pad_sequence(features, batch_first=True), lengths, list(texts), list(paths)


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, id_to_char: dict[int, str], blank_id: int = 0) -> list[str]:
    best_ids = logits.argmax(dim=-1).cpu()
    decoded = []
    for sequence, length in zip(best_ids, lengths.cpu().tolist()):
        output = []
        previous = None
        for token_id in sequence[:length].tolist():
            if token_id != blank_id and token_id != previous:
                output.append(id_to_char[token_id])
            previous = token_id
        decoded.append("".join(output))
    return decoded


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions using Levenshtein DP."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [["match"] * cols for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0] = row
        operation[row][0] = "delete"
    for col in range(1, cols):
        distance[0][col] = col
        operation[0][col] = "insert"
    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                distance[row][col] = distance[row - 1][col - 1]
                operation[row][col] = "match"
            else:
                choices = [
                    (distance[row - 1][col - 1] + 1, "substitute"),
                    (distance[row - 1][col] + 1, "delete"),
                    (distance[row][col - 1] + 1, "insert"),
                ]
                distance[row][col], operation[row][col] = min(choices)
    substitutions = deletions = insertions = 0
    row, col = len(reference), len(hypothesis)
    while row or col:
        action = operation[row][col]
        if action == "substitute":
            substitutions += 1
            row, col = row - 1, col - 1
        elif action == "delete":
            deletions += 1
            row -= 1
        elif action == "insert":
            insertions += 1
            col -= 1
        else:
            row, col = row - 1, col - 1
    return substitutions, deletions, insertions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--streaming", action="store_true", help="evaluate a streaming CTC checkpoint (TinyStreamingConformerCTC)")
    parser.add_argument("--chunk-size", type=int, default=32, help="streaming chunk-size (block frames); only used with --streaming")
    parser.add_argument("--left-context", type=int, default=16, help="streaming left-context frames; only used with --streaming")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples; 0 means all")
    parser.add_argument("--output", type=Path, default=Path("outputs/02_acoustic_encoder"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    id_to_char = {index: char for char, index in vocab.items()}
    if args.streaming:
        model = TinyStreamingConformerCTC(
            len(vocab), chunk_size=args.chunk_size, left_context=args.left_context
        ).to(device)
    else:
        model = TinyConformerCTC(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = AishellEvalDataset(args.data / f"{args.split}.csv")
    if args.limit:
        dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    prediction_path = args.output / f"{args.split}_predictions.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    total_substitutions = total_deletions = total_insertions = total_reference_chars = 0
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "reference", "prediction"])
        writer.writeheader()
        with torch.no_grad():
            for features, lengths, references, paths in tqdm(loader, desc=f"evaluate {args.split}", unit="batch"):
                features = features.to(device)
                frame_steps = torch.arange(features.size(1), device=device).unsqueeze(0)
                padding_mask = frame_steps >= lengths.to(device).unsqueeze(1)
                logits = model(features, padding_mask)
                output_lengths = model.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
                predictions = greedy_decode(logits, output_lengths, id_to_char)
                for path, reference, prediction in zip(paths, references, predictions):
                    substitutions, deletions, insertions = edit_counts(reference, prediction)
                    total_substitutions += substitutions
                    total_deletions += deletions
                    total_insertions += insertions
                    total_reference_chars += len(reference)
                    writer.writerow({"path": path, "reference": reference, "prediction": prediction})

    errors = total_substitutions + total_deletions + total_insertions
    metrics = {
        "split": args.split,
        "examples": len(dataset),
        "reference_characters": total_reference_chars,
        "substitutions": total_substitutions,
        "deletions": total_deletions,
        "insertions": total_insertions,
        "edit_distance": errors,
        "cer": errors / max(total_reference_chars, 1),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
    }
    metrics_path = args.output / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"predictions_saved: {prediction_path}")
    print(f"metrics_saved: {metrics_path}")


if __name__ == "__main__":
    main()
