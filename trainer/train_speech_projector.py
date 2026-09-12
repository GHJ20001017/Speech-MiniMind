"""Train a minimal Speech-MiniMind projector bridge on AISHELL CSV or JSONL data.

AISHELL (`*.csv`) keeps legacy format:

    path,text

JSONL instruction data (`*.jsonl`) should provide:

    audio,instruction,answer

`Aishell` samples still use a fixed prompt (from --prompt).
Instruction JSONL uses `instruction` as prompt and `answer` as target text.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.speech_dataset import SpeechAugmentConfig, SpeechWaveformAugmenter
from model.frozen_encoder import FrozenSpeechEncoder, build_frozen_encoder  # noqa: E402
from model import ddp_utils  # noqa: E402
from model.minimind_adapter import forward_inputs_embeds, load_minimind, token_embeddings  # noqa: E402
from model.speech_projector import SpeechProjector  # noqa: E402
from scripts.analyze_audio import read_wav  # noqa: E402

SAMPLE_RATE = 16000


class AishellCSVDataset(Dataset):
    def __init__(self, manifest: Path, prompt: str, augmenter=None) -> None:
        with manifest.open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.prompt = prompt
        self.manifest = manifest
        self.augmenter = augmenter or SpeechWaveformAugmenter(SpeechAugmentConfig())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str, str]:
        row = self.rows[index]
        audio = Path(row["path"])
        audio_array, rate = read_wav(audio)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, got {rate} (path={audio})")
        waveform = self.augmenter(audio_array, rate)
        text = "".join(row["text"].split())
        return torch.from_numpy(waveform.astype(np.float32)), rate, self.prompt, text, row["path"]


class InstructionJSONLDataset(Dataset):
    def __init__(self, manifest: Path, prompt_fallback: str, augmenter=None) -> None:
        rows: list[dict[str, str]] = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                audio = str(record.get("audio", "")).strip()
                prompt = str(record.get("instruction", "")).strip()
                answer = str(record.get("answer", "")).strip()
                if not audio or not answer:
                    continue
                rows.append(
                    {
                        "audio": audio,
                        "prompt": prompt or prompt_fallback,
                        "answer": answer,
                    }
                )
        self.rows = rows
        self.manifest = manifest
        self.augmenter = augmenter or SpeechWaveformAugmenter(SpeechAugmentConfig())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str, str]:
        row = self.rows[index]
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = self.manifest.parent / audio_path
        audio_array, rate = read_wav(audio_path)
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz waveform, got {rate} (path={audio_path})")
        waveform = self.augmenter(audio_array, rate)
        return torch.from_numpy(waveform.astype(np.float32)), rate, row["prompt"], row["answer"], str(audio_path)


def collate(batch: list[tuple[torch.Tensor, int, str, str, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    waveforms, rates, prompts, texts, paths = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in waveforms], dtype=torch.long)
    return pad_sequence(waveforms, batch_first=True), lengths, list(prompts), list(texts), list(paths)


def make_batch_embeddings(
    model,
    tokenizer,
    projected: torch.Tensor,
    projected_lengths: torch.Tensor,
    prompts: list[str],
    texts: list[str],
    device: torch.device,
    max_length: int,
    max_speech_tokens: int,
):
    sequences: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for speech, speech_length, prompt, text in zip(projected, projected_lengths.tolist(), prompts, texts):
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


def encode_batch(encoder, waveforms, lengths, device, feature_augment=False):
    """Encode one batch through the unified FrozenSpeechEncoder interface."""
    with torch.no_grad():
        acoustic, acoustic_lengths = encoder.encode(
            waveforms.to(device), lengths.to(device), SAMPLE_RATE,
            feature_augment=feature_augment,
        )
    return acoustic, acoustic_lengths


def load_or_encode_batch(encoder, waveforms, lengths, paths, cache_dir, device, feature_augment=False):
    """Return cached (hidden, hidden_lengths) per sample, else encode and write.

    Cache files are ``{cache_dir}/<sha1(path)>.pt``. We always recompute in the
    same batch when any member is missing, then persist what we computed.
    """
    import hashlib

    cache = {}
    to_encode = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        key = hashlib.sha1(path.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{key}.pt"
        if cache_file.exists():
            cache[i] = torch.load(cache_file, weights_only=True, map_location=device)
        else:
            to_encode.append(i)
    if to_encode:
        idx = torch.tensor(to_encode, dtype=torch.long, device=waveforms.device)
        sub = torch.index_select(waveforms, 0, idx)
        sub_lens = torch.index_select(lengths, 0, idx)
        hidden, hidden_lengths = encode_batch(encoder, sub, sub_lens, device, feature_augment)
        for pos, i in enumerate(to_encode):
            key = hashlib.sha1(paths[i].encode()).hexdigest()[:16]
            cache_file = cache_dir / f"{key}.pt"
            entry = (hidden[pos].cpu(), hidden_lengths[pos].cpu())
            torch.save(entry, cache_file)
            cache[i] = tuple(v.to(device) for v in entry)
    # reassemble in original order
    batch_hidden = torch.stack([cache[i][0] for i in range(len(paths))], dim=0)
    batch_lengths = torch.stack([cache[i][1] for i in range(len(paths))], dim=0)
    return batch_hidden, batch_lengths


def run_epoch(args, encoder, projector, lm, tokenizer, loader, device, optimizer=None, cache_dir=None, sampler=None, epoch=0):
    training = optimizer is not None
    if sampler is not None:
        ddp_utils.set_epoch(sampler, epoch)
    projector.train(training)
    total_loss = 0.0
    for waveforms, lengths, prompts, texts, paths in tqdm(loader, desc="train" if training else "dev", unit="batch", disable=not ddp_utils.is_main()):
        if cache_dir is not None:
            acoustic, acoustic_lengths = load_or_encode_batch(
                encoder, waveforms, lengths, paths, cache_dir, device,
                feature_augment=args.augment_mel and training,
            )
        else:
            acoustic, acoustic_lengths = encode_batch(
                encoder, waveforms, lengths, device,
                feature_augment=args.augment_mel and training,
            )
        projected = projector(acoustic)
        projected_lengths = projector.output_lengths(acoustic_lengths).clamp_max(projected.size(1))
        inputs_embeds, attention_mask, labels = make_batch_embeddings(
            lm,
            tokenizer,
            projected,
            projected_lengths,
            prompts,
            texts,
            device,
            args.max_length,
            args.max_speech_tokens,
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
            if args.wandb and ddp_utils.is_main():
                wandb.log({"train/loss_step": loss.detach().item()})
        total_loss += loss.detach().item()
    return ddp_utils.all_reduce_mean(total_loss / max(len(loader), 1))


def resolve_dataset(
    data_root: Path,
    split: str,
    data_format: str,
    prompt_fallback: str,
    augmenter=None,
) -> tuple[Dataset, str]:
    csv_path = data_root / f"{split}.csv"
    jsonl_path = data_root / f"{split}.jsonl"

    if data_root.is_file():
        if split != data_root.stem and not data_root.name.startswith(split):
            raise ValueError(
                f"--data 为单文件时需要文件名与 split 对齐（当前 split={split}，data={data_root}）"
            )
        if data_root.suffix.lower() == ".csv":
            return AishellCSVDataset(data_root, prompt_fallback, augmenter), "csv"
        if data_root.suffix.lower() == ".jsonl":
            return InstructionJSONLDataset(data_root, prompt_fallback, augmenter), "jsonl"
        raise ValueError(f"--data 单文件仅支持 .csv/.jsonl: {data_root}")

    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data path not found: {data_root}")

    if data_format == "csv":
        if not csv_path.exists():
            raise FileNotFoundError(f"csv split不存在: {csv_path}")
        return AishellCSVDataset(csv_path, prompt_fallback, augmenter), "csv"

    if data_format == "jsonl":
        if not jsonl_path.exists():
            raise FileNotFoundError(f"jsonl split不存在: {jsonl_path}")
        return InstructionJSONLDataset(jsonl_path, prompt_fallback, augmenter), "jsonl"

    # auto mode: prefer legacy csv for backward compatibility
    if csv_path.exists():
        return AishellCSVDataset(csv_path, prompt_fallback, augmenter), "csv"
    if jsonl_path.exists():
        return InstructionJSONLDataset(jsonl_path, prompt_fallback, augmenter), "jsonl"
    raise FileNotFoundError(
        f"未发现数据文件: {data_root / 'train.csv'}/{data_root / 'train.jsonl'} (split={split})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--encoder-type", choices=("conformer", "paraformer"), default="conformer",
                        help="frozen acoustic encoder backend (conformer=offline default, paraformer=FunASR streaming)")
    parser.add_argument("--encoder-checkpoint", type=Path, default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--paraformer-model", default=None,
                        help="FunASR model id/dir for --encoder-type paraformer (default ModelScope iic/...-online)")
    parser.add_argument("--hidden-cache", type=Path, default=None,
                        help="directory to cache per-utterance encoder hidden states across epochs (.pt, keyed by sha1(path))")
    parser.add_argument("--minimind-model", type=Path, required=True, help="local Transformers-format MiniMind model directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/03_speech_minimind"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--prompt", default="请将这段语音转写为文字：")
    parser.add_argument("--data-format", choices=("auto", "csv", "jsonl"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="limit examples for a quick smoke test")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False,
                        help="apply random waveform augmentation inside the training dataset")
    parser.add_argument("--augment-mel", action=argparse.BooleanOptionalAction, default=False,
                        help="apply SpecAugment masks after the encoder frontend")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    ddp_utils.setup()
    device = ddp_utils.device()
    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed. Run: python -m pip install -r requirements.txt")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    encoder = build_frozen_encoder(
        encoder_type=args.encoder_type,
        checkpoint=str(args.encoder_checkpoint) if args.encoder_type == "conformer" else None,
        model_id=args.paraformer_model,
        device=device,
    )
    acoustic_dim = encoder.output_dim

    lm, tokenizer = load_minimind(args.minimind_model, device)
    for parameter in lm.parameters():
        parameter.requires_grad_(False)

    llm_dim = int(lm.config.hidden_size)
    projector = ddp_utils.wrap(SpeechProjector(acoustic_dim, llm_dim).to(device))
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    train_augmenter = SpeechWaveformAugmenter(SpeechAugmentConfig(enabled=args.augment))
    dev_augmenter = SpeechWaveformAugmenter(SpeechAugmentConfig(enabled=False))
    train_set, train_format = resolve_dataset(args.data, "train", args.data_format, args.prompt, train_augmenter)
    dev_set, dev_format = resolve_dataset(args.data, "dev", args.data_format, args.prompt, dev_augmenter)
    if train_format != dev_format:
        raise ValueError(f"train/dev 数据格式不一致: train={train_format}, dev={dev_format}")

    if args.limit:
        train_set.rows = train_set.rows[: args.limit]
        dev_set.rows = dev_set.rows[: min(args.limit, len(dev_set))]

    train_sampler = ddp_utils.make_sampler(train_set, shuffle=True)
    dev_sampler = ddp_utils.make_sampler(dev_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, sampler=dev_sampler, collate_fn=collate)

    if ddp_utils.is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(
            json.dumps(vars(args) | {"device": str(device), "llm_hidden_size": llm_dim}, default=str, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writeheader()

    if ddp_utils.is_main():
        print(f"device: {device}")
        print(f"acoustic_dim: {acoustic_dim} (frame_shift_ms={encoder.output_frame_shift_ms})")
        print(f"projector_parameters: {sum(p.numel() for p in projector.parameters()):,}")
        print(f"data_format: {train_format}")
        print(f"train_set_rows: {len(train_set)}")
        print(f"dev_set_rows: {len(dev_set)}")

    effective_cache = None if args.augment or args.augment_mel else args.hidden_cache
    if (args.augment or args.augment_mel) and args.hidden_cache and ddp_utils.is_main():
        print("augmentation enabled: disabling hidden-cache so each epoch gets fresh augmentation")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(args, encoder, projector, lm, tokenizer, train_loader, device, optimizer,
                               cache_dir=effective_cache, sampler=train_sampler, epoch=epoch)
        with torch.no_grad():
            dev_loss = run_epoch(args, encoder, projector, lm, tokenizer, dev_loader, device,
                                 cache_dir=args.hidden_cache)
        if ddp_utils.is_main():
            torch.save({"projector": ddp_utils.unwrap(projector).state_dict(), "epoch": epoch, "llm_hidden_size": llm_dim}, args.output / f"projector_epoch_{epoch:03d}.pt")
            with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "dev_loss"]).writerow(
                    {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "dev_loss": f"{dev_loss:.6f}"}
                )
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")
        if args.wandb and ddp_utils.is_main():
            wandb.log({"train/loss": train_loss, "dev/loss": dev_loss, "epoch": epoch})
    if args.wandb and ddp_utils.is_main():
        wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
