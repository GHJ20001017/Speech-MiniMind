#!/usr/bin/env python3
"""Align SenseVoice input to a format-v5 Thinker/Talker with Mimi output.

Single-device or torchrun DDP training; no LoRA or optimizer resume.
"""
import argparse
# Local package imports below intentionally follow the direct-script path setup.
# ruff: noqa: E402
import random
import json
from pathlib import Path
import sys
import os
from datetime import timedelta
from contextlib import nullcontext
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.thinker_talker_s2s import ThinkerTalkerS2SDataset, prepare_s2s_batch, scheduled_sampling
from model.speech_input import SpeechInputModel, load_s2s_checkpoint, save_s2s_checkpoint, load_encoder
from trainer.train_audio_multitask import load_multitask_checkpoint, multitask_loss, validate_init_checkpoint


def batch_loss(model, batch, device, loss_chunk, *, return_components=False):
    """Preserve the training objective; optionally expose detached terms.

    audio_loss is the actual multitask weighted term (eight codebook means,
    STOP numerator weight 10), not unweighted CE. text_loss + audio_loss = loss.
    """
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    targets = [batch.pop("text_targets"), batch.pop("audio_targets")]
    out = model.forward_streams(**batch)
    result = multitask_loss(model, out.text_hidden, *targets, loss_chunk,
                            audio_hidden=out.audio_hidden, return_components=return_components)
    return (result[0], result[2]) if return_components else result[0]


def preflight_dataset(dataset, tokenizer, max_length, model):
    """Reject invalid pairs before any optimizer step or run-directory creation."""
    for index in range(len(dataset)):
        try:
            for modality in ("audio", "text", "audio_text", "text_audio"):
                prepare_s2s_batch(model, tokenizer, [dataset[index]], max_length, modality=modality)
        except (ValueError, OSError, KeyError) as error:
            raise ValueError(f"{dataset.manifest}: sample {index + 1}: {error}") from error


def supervised_samples(batch):
    """Shifted next-token supervision, kept per row to avoid cross-row signals."""
    return ((batch["text_targets"][:, 1:] != -100).flatten(1).any(dim=1)
            | (batch["audio_targets"][:, 1:] != -1).flatten(1).any(dim=1))


def supervised(batch):
    return bool(supervised_samples(batch).any())


def speech_supervised_count(batch):
    mask = batch.get("speech_mask")
    if mask is None:
        return 0
    return int((supervised_samples(batch) & mask.any(dim=1)).sum())


class LossForward(torch.nn.Module):
    """Keep custom streams AND trainable loss heads inside DDP.forward.

    A zero anchor supplies a backward hook on text-only/unsupervised ranks.
    The caller clears synthetic-only gradients before AdamW can decay them.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, device, loss_chunk, *, return_components=False):
        anchor = next(self.model.frontend.parameters()).reshape(-1)[0] * 0
        metrics = {"text_loss": anchor.detach(), "audio_loss": anchor.detach()}
        if supervised(batch):
            if return_components:
                loss, metrics = batch_loss(self.model, batch, device, loss_chunk, return_components=True)
            else:
                loss = batch_loss(self.model, batch, device, loss_chunk)
        else:
            loss = anchor.detach()
        active = torch.tensor(int(loss.requires_grad), device=device)
        result = (loss + anchor, active)
        return (*result, metrics) if return_components else result


def collective_sum(values, device):
    result = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(result)
    return result.tolist()


def synchronized_call(function):
    """Propagate local IO/preparation/forward failures before next collective."""
    error, result = None, None
    try:
        result = function()
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}")
        if any(errors):
            raise RuntimeError(f"S2S distributed failure: {errors}") from error
    elif error is not None:
        raise error
    return result


class WandbLogger:
    """Optional rank-zero scalar logging; external failures abort all ranks together.

    No login, media, checkpoint uploads or environment/path config collection.
    W&B's history index is separate from the optimizer's global_step: dev and
    epoch summaries never overwrite the final training-step observation.
    """
    def __init__(self, args):
        self.enabled = getattr(args, "wandb", False)
        self.interval = getattr(args, "wandb_log_interval", 1)
        self.run = None
        self.is_main = not dist.is_initialized() or dist.get_rank() == 0
        self.args = args

    def start(self):
        if not self.enabled:
            return
        def initialize():
            if not self.is_main:
                return
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("--wandb requires wandb; install it with: python -m pip install wandb") from exc
            # Explicit safe scalar allowlist; never serialize data paths or credentials.
            keys = ("tuning", "epochs", "batch_size", "lr", "max_seq_len", "loss_chunk",
                    "grad_clip", "seed", "wandb_log_interval")
            config = {key: getattr(self.args, key) for key in keys if hasattr(self.args, key)}
            config["gradient_accumulation_steps"] = getattr(self.args, "gradient_accumulation_steps", 1)
            config["world_size"] = dist.get_world_size() if dist.is_initialized() else 1
            config["effective_batch_size"] = (getattr(self.args, "batch_size", 1)
                                              * config["world_size"] * config["gradient_accumulation_steps"])
            self.run = wandb.init(project=getattr(self.args, "wandb_project", "Speech-MiniMind"),
                                  name=getattr(self.args, "wandb_name", None), config=config,
                                  settings=wandb.Settings(init_timeout=60))
            self.run.define_metric("*", step_metric="global_step")
        synchronized_call(initialize)

    def log(self, values):
        if self.enabled:
            synchronized_call(lambda: self.run.log(values) if self.is_main else None)

    def finish(self):
        if self.enabled:
            synchronized_call(lambda: self.run.finish() if self.run is not None else None)


def reduced_metrics(values):
    """Sum detached [loss, text, weighted audio, supervised rank-batches]."""
    values = values.clone()
    if dist.is_initialized():
        dist.all_reduce(values)
    total, text, audio, count = values.tolist()
    return {"loss": total / count, "text_loss": text / count,
            "audio_loss": audio / count} if count else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="comma-separated raw A2A parquet paths")
    parser.add_argument("--dev-data", default=None, help="optional held-out parquet paths")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=None, help="local SenseVoiceSmall directory; fresh default ../model/SenseVoiceSmall")
    parser.add_argument("--tuning", choices=("audio_proj", "all"), default="audio_proj")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--loss-chunk", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=1,
                        help="log every N synchronized optimizer updates")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.gradient_accumulation_steps,
            args.max_seq_len, args.loss_chunk, args.wandb_log_interval) < 1
            or not 0 < args.lr < float("inf") or not 0 < args.grad_clip < float("inf")):
        parser.error("counts, learning rate and gradient clip must be positive and finite")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl" if args.device.startswith("cuda") else "gloo",
                                timeout=timedelta(minutes=5))
    try:
        train(args)
    finally:
        if distributed:
            dist.destroy_process_group()


def train(args):
    logger = WandbLogger(args)
    try:
        _train(args, logger)
    finally:
        logger.finish()


def _train(args, logger):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    accumulation = getattr(args, "gradient_accumulation_steps", 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("gradient accumulation steps must be positive")
    def check_output():
        if args.output.exists():
            raise ValueError("--output already exists; choose a new run directory")
    # Preserve argparse's single-device CLI error contract.
    if world == 1 and args.output.exists():
        raise SystemExit("--output already exists; choose a new run directory")
    synchronized_call(check_output)
    logger.start()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    def load_model():
        metadata = validate_init_checkpoint(args.init_checkpoint)
        if (metadata.get("task") == "s2s" or (args.init_checkpoint / "s2s_metadata.json").exists()
                or (args.init_checkpoint / "speech_frontend.pt").exists()):
            model, tokenizer = load_s2s_checkpoint(args.init_checkpoint, args.device, encoder_path=args.encoder)
        else:
            base, tokenizer = load_multitask_checkpoint(args.init_checkpoint, args.device)
            encoder_path = args.encoder or Path("../model/SenseVoiceSmall")
            model = SpeechInputModel(base, load_encoder(encoder_path, args.device), encoder_path)
        model.set_tuning(args.tuning)
        if args.max_seq_len > model.config.max_position_embeddings:
            raise ValueError("--max-seq-len exceeds model context")
        return model, tokenizer
    if rank == 0:
        print("loading model...", flush=True)
    model, tokenizer = synchronized_call(load_model)
    if rank == 0:
        print("loaded model", flush=True)
    loaders = {}
    for split, path in (("train", args.data), ("dev", args.dev_data)):
        if path is None:
            continue
        if rank == 0:
            print(f"loading {split} data...", flush=True)
        dataset = synchronized_call(lambda: ThinkerTalkerS2SDataset(
            path, tokenizer, args.max_seq_len, training=split == "train"))
        if rank == 0:
            print(f"loaded {split}: {len(dataset)} parquet rows", flush=True)
        sampler = (DistributedSampler(dataset, shuffle=split == "train", seed=args.seed)
                   if world > 1 else None)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=split == "train" and sampler is None,
                                    sampler=sampler, collate_fn=list)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    forward = LossForward(model)
    if world > 1:
        forward = DistributedDataParallel(
            forward, device_ids=[torch.device(args.device).index] if args.device.startswith("cuda") else None,
            find_unused_parameters=True, broadcast_buffers=False)

    def create_output():
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    synchronized_call(create_output)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        seed = args.seed + (epoch - 1) * world + rank
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        report = {"epoch": epoch}
        epoch_metrics = {}
        for split, loader in loaders.items():
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch - 1)
            model.train(split == "train")
            total, batches, updates = 0.0, 0, 0
            metric_total = torch.zeros(4, device=args.device, dtype=torch.float64)
            progress = synchronized_call(lambda: tqdm(
                total=len(loader), desc=f"{split} epoch {epoch}/{args.epochs}",
                file=sys.stdout, mininterval=0, miniters=1, dynamic_ncols=True,
            ) if rank == 0 else None)

            def show_step(index, values, status):
                if rank != 0:
                    return
                fields = " ".join(f"{key}={values[key]:.6f}" if key in values else f"{key}=n/a"
                                  for key in ("loss", "text_loss", "audio_loss"))
                progress.update(1)
                tqdm.write(f"{split} epoch={epoch}/{args.epochs} batch={index}/{len(loader)} "
                           f"global_step={global_step} {fields} "
                           f"lr={optimizer.param_groups[0]['lr']:.6g} {status}", file=sys.stdout)
                sys.stdout.flush()

            with (progress if progress is not None else nullcontext()), torch.set_grad_enabled(split == "train"):
                iterator = iter(loader)
                for batch_index in range(1, len(loader) + 1):
                    if split == "train" and (batch_index - 1) % accumulation == 0:
                        optimizer.zero_grad(set_to_none=True)
                        window_metrics = torch.zeros_like(metric_total)
                        window_count = 0
                        window_active = window_speech = False
                    raw = synchronized_call(lambda: next(iterator))
                    batch = synchronized_call(lambda: prepare_s2s_batch(
                        model, tokenizer, raw, args.max_seq_len, mixture=split == "train"))
                    valid = supervised(batch)
                    count, = collective_sum([int(valid)], args.device)
                    if split == "train":
                        batch = synchronized_call(lambda: scheduled_sampling(batch, tokenizer))
                    loss, active, components = synchronized_call(lambda: forward(
                        batch, args.device, args.loss_chunk, return_components=True))
                    metrics = torch.stack((loss.detach(), components["text_loss"].detach(),
                                           components["audio_loss"].detach(), loss.new_tensor(int(valid)))).double()
                    if not valid:
                        metrics.zero_()
                    metric_total += metrics
                    # One shared reduction per supervised batch, independent of W&B cadence.
                    step_values = reduced_metrics(metrics)
                    status = "evaluated" if count else "skipped=no_supervision"
                    finite, = collective_sum([int(torch.isfinite(loss))], args.device)
                    if finite != world:
                        raise RuntimeError("nonfinite S2S loss; refusing checkpoint")
                    active_count, speech_count = collective_sum(
                        [int(active), speech_supervised_count(batch) if active else 0],
                        args.device)
                    if split == "train":
                        # Synchronize every microbatch, including empty ranks. DDP
                        # averages ranks; normalize once by the window's valid count.
                        (loss * world).backward()
                        window_count += count
                        window_metrics += metrics
                        window_active |= bool(active_count)
                        window_speech |= bool(speech_count)
                        position = (batch_index - 1) % accumulation + 1
                        window_size = min(accumulation, len(loader) - batch_index + position)
                        status = "accumulating"
                        if position == window_size:
                            if not window_speech:
                                for parameter in model.frontend.parameters():
                                    parameter.grad = None
                            update = window_speech if args.tuning == "audio_proj" else window_active
                            status = "skipped=no_active_gradients" if window_count else "skipped=no_supervision"
                            if update and window_count:
                                for parameter in parameters:
                                    if parameter.grad is not None:
                                        parameter.grad.div_(window_count)
                                synchronized_call(lambda: torch.nn.utils.clip_grad_norm_(
                                    parameters, args.grad_clip, error_if_nonfinite=True))
                                synchronized_call(optimizer.step)
                                updates += 1
                                global_step += 1
                                status = "updated"
                                if logger.enabled and global_step % logger.interval == 0:
                                    values = {f"train/{key}": value
                                              for key, value in reduced_metrics(window_metrics).items()}
                                    logger.log(dict(values, global_step=global_step, epoch=epoch,
                                                    lr=optimizer.param_groups[0]["lr"], updates=global_step,
                                                    epoch_updates=updates))
                        status = f"{status} accumulation={position}/{window_size}"
                    synchronized_call(lambda: show_step(batch_index, step_values, status))
                    batches += int(valid)
            values = reduced_metrics(metric_total)
            if logger.enabled:
                # Train epoch means have separate names from optimizer-step metrics.
                prefix = "train/epoch_" if split == "train" else "dev/"
                epoch_metrics.update({prefix + key: value for key, value in values.items()})
            _, batches = collective_sum([0, batches], args.device)
            total = values.get("loss", 0.0) * batches
            if not batches:
                raise ValueError(f"{split} has no next-token supervision after truncation")
            report[split + "_loss"] = total / batches
            report[split + "_batches"] = int(batches)
            if split == "train":
                report["train_updates"] = updates  # synchronized steps, not world * steps
                if not updates:
                    raise ValueError("train epoch has zero optimizer updates; refusing checkpoint")
        if logger.enabled:
            logger.log(dict(epoch_metrics, global_step=global_step, epoch=epoch,
                            lr=optimizer.param_groups[0]["lr"], updates=global_step,
                            epoch_updates=report["train_updates"]))
        synchronized_call(lambda: save_s2s_checkpoint(
            model, tokenizer, args.output / f"model_epoch_{epoch:03d}", epoch) if rank == 0 else None)
        if rank == 0:
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
