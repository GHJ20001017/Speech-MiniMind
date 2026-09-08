"""Train a minimal Speech-MiniMind bridge on AISHELL-1 transcripts.

The Tiny Conformer and MiniMind are frozen by default. Only SpeechProjector is
trained, making the first multimodal experiment small enough to understand and
run before attempting full-model fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.ctc_model import TinyConformerCTC  # noqa: E402
from model.minimind_adapter import forward_inputs_embeds, load_minimind, token_embeddings  # noqa: E402
from model.speech_projector import SpeechProjector  # noqa: E402
from scripts.evaluate_conformer_ctc import AishellEvalDataset  # noqa: E402


def collate(batch):
    features, texts, paths = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    return pad_sequence(features, batch_first=True), lengths, list(texts), list(paths)


def make_batch_embeddings(
    model,
    tokenizer,
    projected: torch.Tensor,
    projected_lengths: torch.Tensor,
    texts: list[str],
    device: torch.device,
    prompt: str,
    max_length: int,
    max_speech_tokens: int,
):
    sequences, labels = [], []
    for speech, speech_length, text in zip(projected, projected_lengths.tolist(), texts):
        speech = speech[: min(speech_length, max_speech_tokens, max_length - 2)]
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        target_ids = tokenizer(text + (tokenizer.eos_token or ""), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([prompt_ids, target_ids])[: max(max_length - speech.size(0), 1)]
        prompt_count = min(prompt_ids.numel(), ids.numel())
        text_labels = ids.clone()
        text_labels[:prompt_count] = -100
        embeddings = torch.cat([speech, token_embeddings(model, ids.to(device)).squeeze(0)], dim=0)
        label = torch.cat([torch.full((speech.size(0),), -100, device=device, dtype=torch.long), text_labels.to(device)])
        sequences.append(embeddings)
        labels.append(label)
    padded = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.zeros(padded.size(0), padded.size(1), device=device, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        attention_mask[row, : sequence.size(0)] = 1
    return padded, attention_mask, padded_labels


def run_epoch(args, encoder, projector, lm, tokenizer, loader, device, optimizer=None):
    training = optimizer is not None
    projector.train(training)
    total_loss = 0.0
    for features, lengths, texts, _ in tqdm(loader, desc="train" if training else "dev", unit="batch"):
        features, lengths = features.to(device), lengths.to(device)
        frame_steps = torch.arange(features.size(1), device=device).unsqueeze(0)
        padding_mask = frame_steps >= lengths.unsqueeze(1)
        with torch.no_grad():
            acoustic = encoder(features, padding_mask)
        projected = projector(acoustic)
        acoustic_lengths = encoder.subsampled_lengths(lengths).clamp_max(acoustic.size(1))
        projected_lengths = projector.output_lengths(acoustic_lengths).clamp_max(projected.size(1))
        inputs_embeds, attention_mask, labels = make_batch_embeddings(
            lm, tokenizer, projected, projected_lengths, texts, device, args.prompt, args.max_length, args.max_speech_tokens
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
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--encoder-checkpoint", type=Path, default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--minimind-model", type=Path, required=True, help="local Transformers-format MiniMind model directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/03_speech_minimind"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--prompt", default="请将这段语音转写为文字：")
    parser.add_argument("--limit", type=int, default=0, help="limit examples for a quick smoke test")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder_model = TinyConformerCTC(len(checkpoint["vocab"])).to(device)
    encoder_model.load_state_dict(checkpoint["model"])
    encoder_model.eval()
    for parameter in encoder_model.parameters():
        parameter.requires_grad_(False)
    lm, tokenizer = load_minimind(args.minimind_model, device)
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    llm_dim = int(lm.config.hidden_size)
    projector = SpeechProjector(256, llm_dim).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    train_set = AishellEvalDataset(args.data / "train.csv")
    dev_set = AishellEvalDataset(args.data / "dev.csv")
    if args.limit:
        train_set.rows = train_set.rows[: args.limit]
        dev_set.rows = dev_set.rows[: min(args.limit, len(dev_set))]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args) | {"device": str(device), "llm_hidden_size": llm_dim}, default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writeheader()
    print(f"device: {device}")
    print(f"projector_parameters: {sum(p.numel() for p in projector.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(args, encoder_model.encoder, projector, lm, tokenizer, train_loader, device, optimizer)
        with torch.no_grad():
            dev_loss = run_epoch(args, encoder_model.encoder, projector, lm, tokenizer, dev_loader, device)
        torch.save({"projector": projector.state_dict(), "epoch": epoch, "llm_hidden_size": llm_dim}, args.output / f"projector_epoch_{epoch:03d}.pt")
        with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writerow({"epoch": epoch, "train_loss": f"{train_loss:.6f}", "dev_loss": f"{dev_loss:.6f}"})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")


if __name__ == "__main__":
    main()
