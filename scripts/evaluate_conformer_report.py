"""Build a complete Tiny Conformer CTC evaluation report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.ctc_model import TinyConformerCTC  # noqa: E402
from scripts.evaluate_conformer_ctc import (  # noqa: E402
    AishellEvalDataset,
    collate,
    edit_counts,
    greedy_decode,
)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    sample_count: int,
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, object]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint["vocab"]
    id_to_char = {index: char for char, index in vocab.items()}
    model = TinyConformerCTC(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = AishellEvalDataset(data_dir / f"{split}.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    substitutions = deletions = insertions = reference_chars = 0
    audio_seconds = inference_seconds = 0.0
    examples_seen = 0
    examples: list[dict[str, str]] = []
    error_cases: list[dict[str, object]] = []
    with torch.no_grad():
        for features, lengths, references, paths in tqdm(loader, desc=f"{checkpoint_path.name} {split}", unit="batch"):
            features = features.to(device)
            frame_steps = torch.arange(features.size(1), device=device).unsqueeze(0)
            padding_mask = frame_steps >= lengths.to(device).unsqueeze(1)
            start = time.perf_counter()
            logits = model(features, padding_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            output_lengths = model.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
            predictions = greedy_decode(logits, output_lengths, id_to_char)
            audio_seconds += float(lengths.sum()) * 0.01
            for path, reference, prediction in zip(paths, references, predictions):
                sub, delete, insert = edit_counts(reference, prediction)
                substitutions += sub
                deletions += delete
                insertions += insert
                reference_chars += len(reference)
                if len(examples) < sample_count:
                    examples.append({"path": path, "reference": reference, "prediction": prediction})
                if sub + delete + insert:
                    error_cases.append(
                        {
                            "path": path,
                            "reference": reference,
                            "prediction": prediction,
                            "substitutions": sub,
                            "deletions": delete,
                            "insertions": insert,
                            "edit_distance": sub + delete + insert,
                            "cer": (sub + delete + insert) / max(len(reference), 1),
                        }
                    )
                examples_seen += 1
    edit_distance_total = substitutions + deletions + insertions
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "examples": examples_seen,
        "reference_characters": reference_chars,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_distance": edit_distance_total,
        "cer": edit_distance_total / max(reference_chars, 1),
        "parameters": parameters,
        "audio_duration_seconds_estimate": audio_seconds,
        "inference_seconds": inference_seconds,
        "real_time_factor": inference_seconds / max(audio_seconds, 1e-9),
        "audio_seconds_per_inference_second": audio_seconds / max(inference_seconds, 1e-9),
        "device": str(device),
    }, examples, error_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/02_acoustic_encoder"))
    parser.add_argument("--checkpoints", type=Path, default=None, help="directory; defaults to --output")
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="only evaluate the latest N checkpoints; 0 means all")
    parser.add_argument("--samples", type=int, default=20, help="number of reference/prediction samples per split")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoints or args.output
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    final_checkpoint = checkpoint_dir / "tiny_conformer_ctc.pt"
    if final_checkpoint.exists() and final_checkpoint not in checkpoints:
        checkpoints.append(final_checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    if args.max_checkpoints:
        checkpoints = checkpoints[-args.max_checkpoints :]
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    sample_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        for split in splits:
            metrics, examples, errors = evaluate_checkpoint(
                checkpoint, args.data, split, args.batch_size, device, args.samples
            )
            all_metrics.append(metrics)
            sample_rows.extend({"checkpoint": str(checkpoint), "split": split, **row} for row in examples)
            # Keep the most useful cases bounded and easy to inspect.
            errors.sort(key=lambda row: (float(row["cer"]), int(row["edit_distance"])), reverse=True)
            error_rows.extend({"checkpoint": str(checkpoint), "split": split, **row} for row in errors[: args.samples])
            print(f"{checkpoint.name} {split}: CER={metrics['cer']:.4%}, RTF={metrics['real_time_factor']:.4f}")

    comparison_path = args.output / "checkpoint_comparison.csv"
    fields = list(all_metrics[0].keys())
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    samples_path = args.output / "evaluation_samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "split", "path", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(sample_rows)
    errors_path = args.output / "evaluation_errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        error_fields = [
            "checkpoint", "split", "path", "reference", "prediction",
            "substitutions", "deletions", "insertions", "edit_distance", "cer",
        ]
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows(error_rows)
    best_by_split = {}
    for split in splits:
        candidates = [row for row in all_metrics if row["split"] == split]
        best_by_split[split] = min(candidates, key=lambda row: float(row["cer"]))
    report = {
        "checkpoints": len(checkpoints),
        "splits": list(splits),
        "best_by_split": best_by_split,
        "metrics": all_metrics,
        "comparison_csv": str(comparison_path),
        "samples_csv": str(samples_path),
        "errors_csv": str(errors_path),
    }
    report_path = args.output / "evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"comparison_saved: {comparison_path}")
    print(f"samples_saved: {samples_path}")
    print(f"errors_saved: {errors_path}")
    print(f"report_saved: {report_path}")


if __name__ == "__main__":
    main()
