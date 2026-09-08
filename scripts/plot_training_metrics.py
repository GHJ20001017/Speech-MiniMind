"""Plot numeric Tiny Conformer train/dev losses from metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("outputs/02_acoustic_encoder/metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/02_acoustic_encoder/loss_curve.png"))
    args = parser.parse_args()

    with args.metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no metric rows found in {args.metrics}")

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_ctc_loss"]) for row in rows]
    dev_losses = [float(row["dev_ctc_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    axis.plot(epochs, train_losses, marker="o", markersize=4, linewidth=2, label="Train")
    axis.plot(epochs, dev_losses, marker="o", markersize=4, linewidth=2, label="Dev")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("CTC loss")
    axis.set_title("Tiny Conformer CTC loss")
    axis.set_xticks(epochs)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    print(f"loss_curve_saved: {args.output}")


if __name__ == "__main__":
    main()
