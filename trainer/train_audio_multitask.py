"""S2A-only training with a Qwen3 text-only Thinker and independent Mimi Talker.

S2A jointly supervises assistant text and delayed audio. Audio never enters the
Thinker. Start directly from original Qwen3 weights, or optionally warm-start
format-v5 weights and tokenizer via --init-checkpoint; optimizer/scheduler state
is reset. No historical TTS stage is required. Shared-backbone checkpoints are
rejected. General batch and checkpoint helpers remain available for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.multitask_audio_dataset import EmiliaTaskDataset, MultiTaskAudioSample, multitask_collate
from model import ddp_utils
from model.audio_sampling import GENERATION_CONFIG
from model.audio_lm import build_vocab_spec, register_audio_special_tokens
from model.qwen3_adapter import lm_head_module, load_qwen3
from model.qwen3_talker import (Qwen3ThinkerTalker, AUDIO_STOP_ID, AUDIO_PAD_ID,
                                 AUDIO_SPEAKER_ID, AUDIO_INPUT_VOCAB_SIZE,
                                 AUDIO_OUTPUT_VOCAB_SIZE)

TASKS = ("s2a",)
S2A_SEMANTICS = {
    "conditioning": "full_prior_text_chat; no_question_audio_or_speaker_conditioning",
    "supervision": "last_assistant_text_including_template_terminator_and_audio_only",
    "alignment": "audio_target_at_answer_body_onset_plus_1_plus_codebook_q; next_token_shift; first_audio_sees_first_answer_text_token",
    "fusion": "text_only_thinker; projected_intermediate_text_plus_codec_history_in_independent_talker",
    "chat_template": "tokenizer_default_thinking_template; single_render_shared_by_text_and_audio; empty_thinking_removed_with_p0.8",
    "length_policy": "random_assistant_then_backward_fallback_while_render_plus_100_ge_max_seq_len; truncate_then_fixed_pad_to_max_seq_len",
    "scope": "joint_training_and_autoregressive_joint_or_teacher_inference",
}
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
AUDIO_HEAD_SIZE = AUDIO_OUTPUT_VOCAB_SIZE
AUDIO_STOP_WEIGHT = 10.0
AUDIO_CODEBOOK_DELAYS = tuple(range(NUM_CODEBOOKS))
AUDIO_SEQUENCE_LAYOUT = "output_step_s_codebook_q_frame_s_minus_q; per_codebook_stop_at_T_plus_q; codes_and_STOP_input_PAD_inactive; undelayed_prompt"
FORMAT_VERSION = 5
DEFAULT_S2A_MAX_SEQ_LEN = 512

def _text_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _audio_stream(codes: torch.Tensor) -> list[list[int]]:
    """Return one local-code vector ``[q0..q7]`` per audio frame."""
    if codes.dim() != 2 or codes.size(0) != NUM_CODEBOOKS:
        raise ValueError(f"expected [{NUM_CODEBOOKS}, frames] Mimi codes, got {tuple(codes.shape)}")
    if codes.is_floating_point() or codes.is_complex() or codes.dtype == torch.bool:
        raise ValueError("Mimi codes must be integer codec indices")
    if ((codes < 0) | (codes >= CODEBOOK_SIZE)).any():
        raise ValueError(f"raw Mimi codes must be in 0..{CODEBOOK_SIZE - 1}; special IDs are added by the sequence builder")
    return codes.transpose(0, 1).tolist()


def _empty_audio_frame() -> list[int]:
    return [-1] * NUM_CODEBOOKS


def _delayed_audio_output(codes: torch.Tensor):
    """Targets and code/STOP feedback for steps 0..T+7, including final STOP.

    A target at row r is predicted from hidden state r-1. Thus codebook q
    can condition on lower codebooks of the same frame, never on itself.
    STOP is fed back at the next prediction; inactive inputs use PAD.
    """
    frames = _audio_stream(codes)
    targets, inputs = [], []
    for step in range(len(frames) + max(AUDIO_CODEBOOK_DELAYS) + 1):
        target, audio_input = _empty_audio_frame(), [AUDIO_PAD_ID] * NUM_CODEBOOKS
        for q, delay in enumerate(AUDIO_CODEBOOK_DELAYS):
            frame = step - delay
            if 0 <= frame < len(frames):
                target[q] = audio_input[q] = frames[frame][q]
            elif frame == len(frames):
                target[q] = audio_input[q] = AUDIO_STOP_ID
        targets.append(target)
        inputs.append(audio_input)
    return targets, inputs


def _s2a_prompt(sample):
    """Consume the fixed-length conversation prompt rendered by the dataset.

    MiniMindT2ADataset owns assistant selection, system/thinking augmentation,
    the backward length fallback and the truncate/pad step, and exposes the
    result as ``prompt_ids``/``answer_start``/``prompt_len``. Reusing exactly
    that render keeps text and audio supervision on one template decision.
    """
    prompt_ids = getattr(sample, "prompt_ids", None)
    answer_start = getattr(sample, "answer_start", None)
    text_start = getattr(sample, "text_start", None)
    prompt_len = getattr(sample, "prompt_len", None)
    if (not isinstance(prompt_ids, list) or not prompt_ids
            or not isinstance(answer_start, int) or not isinstance(prompt_len, int)
            or not isinstance(text_start, int)
            or not 0 <= text_start <= answer_start <= prompt_len <= len(prompt_ids)
            or not all(isinstance(token, int) for token in prompt_ids)):
        raise ValueError("s2a sample is missing dataset-rendered prompt_ids/answer_start/prompt_len")
    if not isinstance(sample.messages, list):
        raise ValueError("s2a sample has no conversation messages")
    return list(prompt_ids), answer_start, prompt_len, text_start


def build_multitask_batch(tokenizer, spec, samples: list[MultiTaskAudioSample], max_length: int):
    """Build text inputs and delayed codec history for independent decoders.

    S2A reuses the dataset's fixed-length render and starts the delayed audio at
    the answer-body onset + 1 (MiniMind-O ``assistant_start + layer_idx + 1``):
    after next-token shifting, q0 sees the first answer token, not the full answer.
    """
    if not samples or max_length < 1:
        raise ValueError("batch must be nonempty and max_length must be positive")
    input_rows, text_rows, audio_rows, audio_input_rows = [], [], [], []
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    for sample in samples:
        row, text_label, audio_label, audio_input = [], [], [], []
        if bos is not None and sample.task != "s2a":
            row.append(bos); text_label.append(-100); audio_label.append(_empty_audio_frame()); audio_input.append(_empty_audio_frame())
        if sample.task == "s2a":
            row, answer_start, prompt_len, text_start = _s2a_prompt(sample)
            targets, feedback = _delayed_audio_output(sample.codes)
            audio_start = answer_start + 1  # Upstream: answer_body_onset + layer_idx + 1.
            text_label = [-100] * text_start + row[text_start:prompt_len] + [-100] * (len(row) - prompt_len)
            placed = [_empty_audio_frame() for _ in range(audio_start)] + targets
            placed_feedback = [_empty_audio_frame() for _ in range(audio_start)] + feedback
            audio_label = placed[:len(row)] + [_empty_audio_frame() for _ in range(len(row) - len(placed))]
            audio_input = (placed_feedback[:len(row)]
                           + [_empty_audio_frame() for _ in range(len(row) - len(placed_feedback))])
        elif sample.task == "asr":
            raise ValueError("ASR is unsupported: the Thinker is text-only and does not consume audio")
        elif sample.task == "tts":
            target = _text_ids(tokenizer, sample.text)
            row += target + [spec.audio_bos_id]
            text_label += [-100] * (len(target) + 1)
            audio_label += [_empty_audio_frame()] * (len(target) + 1)
            audio_input += [_empty_audio_frame()] * (len(target) + 1)
            targets, feedback = _delayed_audio_output(sample.codes)
            row += [pad] * (len(targets) - 1) + [spec.audio_eos_id]
            text_label += [-100] * len(targets)
            audio_label += targets
            audio_input += feedback
        elif sample.task == "audio_lm":
            if sample.prompt_codes is None:
                raise ValueError("audio_lm sample has no prompt")
            prompt_frames = _audio_stream(sample.prompt_codes)
            targets, feedback = _delayed_audio_output(sample.codes)
            row += [spec.audio_bos_id] + [pad] * len(prompt_frames) + [spec.audio_eos_id, spec.audio_bos_id]
            text_label += [-100] * (len(prompt_frames) + 3)
            audio_label += [_empty_audio_frame()] * (len(prompt_frames) + 3)
            audio_input += [_empty_audio_frame()] + prompt_frames + [_empty_audio_frame(), _empty_audio_frame()]
            row += [pad] * (len(targets) - 1) + [spec.audio_eos_id]
            text_label += [-100] * len(targets)
            audio_label += targets
            audio_input += feedback
        else:
            raise ValueError(f"unknown task: {sample.task}")
        if len(row) > max_length:
            raise ValueError(
                f"{sample.task} sequence length {len(row)} exceeds max_length={max_length}; "
                "model context limit exceeded (S2A truncates inside the dataset; other tasks never crop supervision)"
            )
        if sample.task != "s2a" and not any(x != -100 for x in text_label[1:]) and not any(
            x != -1 for frame in audio_label[1:] for x in frame
        ):
            raise ValueError(f"{sample.task} sample has no next-token supervision")
        lengths = {"row": len(row), "text": len(text_label), "audio": len(audio_label), "audio_input": len(audio_input)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{sample.task} batch fields have inconsistent lengths: {lengths}")
        input_rows.append(row); text_rows.append(text_label); audio_rows.append(audio_label); audio_input_rows.append(audio_input)
    length = max(map(len, input_rows))
    inputs = torch.full((len(samples), length), pad, dtype=torch.long)
    text_targets = torch.full_like(inputs, -100)
    audio_targets = torch.full((len(samples), length, NUM_CODEBOOKS), -1, dtype=torch.long)
    audio_inputs = torch.full_like(audio_targets, -1)
    mask = torch.zeros_like(inputs)
    for i, (row, text, audio, audio_input) in enumerate(zip(input_rows, text_rows, audio_rows, audio_input_rows)):
        n = len(row)
        inputs[i, :n] = torch.tensor(row)
        text_targets[i, :n] = torch.tensor(text)
        audio_targets[i, :n] = torch.tensor(audio)
        audio_inputs[i, :n] = torch.tensor(audio_input)
        mask[i, :n] = 1
    audio_inputs.masked_fill_(audio_inputs == -1, AUDIO_PAD_ID)
    return inputs, text_targets, audio_targets, audio_inputs, mask


def _head_logits(head, hidden):
    if torch.is_grad_enabled() and hidden.requires_grad:
        return checkpoint(head.__call__, hidden, use_reentrant=False)
    return head(hidden)


def multitask_loss(model, hidden, text_targets, audio_targets, chunk_size, *, audio_hidden,
                   return_components=False):
    """Text mean CE + mean of eight codebook means; stop has numerator weight 10."""
    h = hidden[:, :-1].reshape(-1, hidden.size(-1))
    if audio_hidden.shape != hidden.shape:
        raise ValueError("text and audio hidden states must have identical shapes")
    ah = audio_hidden[:, :-1].reshape(-1, audio_hidden.size(-1))
    text = text_targets[:, 1:].reshape(-1)
    audio = audio_targets[:, 1:].reshape(-1, NUM_CODEBOOKS)
    text_count = int((text != -100).sum())
    audio_count = int((audio != -1).sum())
    if not text_count + audio_count:
        raise ValueError("batch has no next-token supervision")
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    text_loss = torch.zeros_like(total)
    audio_loss = torch.zeros_like(total)
    chunk = max(1, h.size(0)) if chunk_size <= 0 else chunk_size
    if text_count:
        selected = text != -100
        hs, targets = h[selected], text[selected]
        text_head = lm_head_module(model.thinker)
        for start in range(0, text_count, chunk):
            component = F.cross_entropy(
                _head_logits(text_head, hs[start:start + chunk]),
                targets[start:start + chunk], reduction="sum",
            ) / text_count
            total = total + component
            if return_components:
                text_loss = text_loss + component.detach()
    for codebook, head in enumerate(model.audio_streams.heads):
        selected = audio[:, codebook] != -1
        count = int(selected.sum())
        if count:
            hs, targets = ah[selected], audio[selected, codebook]
            for start in range(0, count, chunk):
                labels = targets[start:start + chunk]
                ce = F.cross_entropy(_head_logits(head, hs[start:start + chunk]), labels, reduction="none")
                weights = torch.where(labels == AUDIO_STOP_ID, AUDIO_STOP_WEIGHT, 1.0)
                component = (ce * weights).sum() / count / NUM_CODEBOOKS
                total = total + component
                if return_components:
                    audio_loss = audio_loss + component.detach()
        else:
            # Keep unused audio heads in the graph without adding supervision.
            for start in range(0, h.size(0), chunk):
                logits = _head_logits(head, ah[start:start + chunk])
                total = total + (logits * 0.0).sum()
    if return_components:
        return total, text_count + audio_count, {"text_loss": text_loss, "audio_loss": audio_loss}
    return total, text_count + audio_count


def perturb_audio_history(audio_inputs, audio_targets, probability):
    """Upstream corruption: shifted supervised times replace all eight cells.

    Labels are stored unshifted here, so input row r uses label row r+1.
    The final input has no prediction target and is never corrupted.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("history noise probability must be in [0, 1]")
    if probability == 0:
        return audio_inputs
    eligible = torch.zeros(audio_inputs.shape[:-1], dtype=torch.bool, device=audio_inputs.device)
    eligible[:, :-1] = (audio_targets[:, 1:] != -1).any(dim=-1)
    selected = eligible & (torch.rand(eligible.shape, device=audio_inputs.device) < probability)
    noise = torch.randint(AUDIO_INPUT_VOCAB_SIZE, audio_inputs.shape,
                          device=audio_inputs.device, dtype=audio_inputs.dtype)
    return torch.where(selected.unsqueeze(-1), noise, audio_inputs)


def perturb_text_history(inputs, text_targets, probability, vocab_size, image_token_id=None):
    """Independent shifted text mask; protect existing image marker inputs."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("history noise probability must be in [0, 1]")
    if probability == 0:
        return inputs
    eligible = torch.zeros_like(inputs, dtype=torch.bool)
    eligible[:, :-1] = text_targets[:, 1:] != -100
    if image_token_id is not None:
        eligible &= inputs != image_token_id
    selected = eligible & (torch.rand(inputs.shape, device=inputs.device) < probability)
    noise = torch.randint(vocab_size, inputs.shape, device=inputs.device, dtype=inputs.dtype)
    return torch.where(selected, noise, inputs)


class MultitaskForward(torch.nn.Module):
    """Keep embedding, decoder, heads and loss inside the DDP forward boundary."""

    def __init__(self, backbone, chunk_size, history_noise_prob=0.05,
                 text_vocab_size=None, image_token_id=None):
        super().__init__()
        if not 0.0 <= history_noise_prob <= 1.0:
            raise ValueError("history noise probability must be in [0, 1]")
        self.backbone = backbone
        self.chunk_size = chunk_size
        self.history_noise_prob = history_noise_prob
        self.text_vocab_size = text_vocab_size
        self.image_token_id = image_token_id

    def forward(self, inputs, text_targets, audio_targets, audio_inputs, attention, *,
                return_components=False):
        if self.training and self.history_noise_prob:
            audio_inputs = perturb_audio_history(audio_inputs, audio_targets, self.history_noise_prob)
            if (text_targets[:, 1:] != -100).any():
                inputs = perturb_text_history(inputs, text_targets, self.history_noise_prob,
                                              self.text_vocab_size or self.backbone.get_input_embeddings().num_embeddings,
                                              self.image_token_id)
        streams = self.backbone.forward_streams(inputs, audio_inputs, attention_mask=attention)
        return multitask_loss(self.backbone, streams.text_hidden, text_targets, audio_targets,
                              self.chunk_size, audio_hidden=streams.audio_hidden,
                              return_components=return_components)


def make_loader(dataset, batch_size, device, train, sampler, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                      collate_fn=multitask_collate, num_workers=num_workers, pin_memory=device.type == "cuda",
                      persistent_workers=False)


def run_epoch(args, model, tokenizer, spec, loaders, device, optimizer=None, scheduler=None,
              epoch=0, loss_ema=None, *, return_components=False):
    """Optionally return (joint mean, EMA mapping, component means).

    All metrics use equal microbatch means, then equal rank means, just like
    the joint loss. EMA advances once per optimizer update, including the tail.
    The default API retains the original scalar EMA and two-value result.
    """
    training = optimizer is not None
    model.train(training)
    if len(loaders) != 1:
        raise ValueError("run_epoch requires exactly one task loader")
    task, loader = next(iter(loaders.items()))
    loader.dataset.set_epoch(epoch)
    ddp_utils.set_epoch(loader.sampler if hasattr(loader.sampler, "set_epoch") else None, epoch)
    steps = len(loader)
    if not steps:
        raise ValueError(f"{task} loader is empty")
    names = ("joint_loss", "text_loss", "audio_loss") if return_components else ("joint_loss",)
    totals = dict.fromkeys(names, 0.0)
    window = dict.fromkeys(names, 0.0)
    emas = ({name: (loss_ema or {}).get(name) for name in names}
            if return_components else {"joint_loss": loss_ema})
    accum = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, samples in enumerate(tqdm(loader, desc=f"{task} {'train' if training else 'dev'}", disable=not ddp_utils.is_main())):
        batch = tuple(tensor.to(device) for tensor in
                      build_multitask_batch(tokenizer, spec, samples, args.max_length))
        with torch.set_grad_enabled(training):
            if return_components:
                loss, _, components = model(*batch, return_components=True)
                values = {name: components[name].detach().item() for name in names[1:]}
            else:
                loss, _ = model(*batch)
                values = {}
            values["joint_loss"] = loss.detach().item()
            for name in names:
                totals[name] += values[name]
            if training:
                for name in names:
                    window[name] += values[name]
                (loss / args.grad_accum_steps).backward()
                accum += 1
                if accum == args.grad_accum_steps or step == steps - 1:
                    if accum < args.grad_accum_steps:
                        scale = args.grad_accum_steps / accum
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                    optimizer.step()
                    if scheduler is not None: scheduler.step()
                    # Every rank participates, regardless of W&B/main-rank status.
                    means = {name: ddp_utils.all_reduce_mean(window[name] / accum) for name in names}
                    if args.loss_ema > 0:
                        for name in names:
                            emas[name] = (means[name] if emas[name] is None else
                                          (1.0 - args.loss_ema) * emas[name] + args.loss_ema * means[name])
                    if args.wandb and ddp_utils.is_main():
                        loss_name = "joint_loss" if task == "s2a" else "audio_loss"
                        metrics = {
                            f"train/{task}/{loss_name}_step": means["joint_loss"],
                            f"train/{task}/{loss_name}_ema": emas["joint_loss"],
                            "train/lr_backbone": optimizer.param_groups[0]["lr"],
                            "train/lr_audio": optimizer.param_groups[1]["lr"],
                            "task": task,
                            "epoch": epoch,
                        }
                        if return_components and task == "s2a":
                            for name in names[1:]:
                                metrics[f"train/{task}/{name}_step"] = means[name]
                                metrics[f"train/{task}/{name}_ema"] = emas[name]
                        wandb.log(metrics)
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    window = dict.fromkeys(names, 0.0)
    means = {name: ddp_utils.all_reduce_mean(totals[name] / steps) for name in names}
    if return_components:
        return means["joint_loss"], emas, {name: means[name] for name in names[1:]}
    return means["joint_loss"], emas["joint_loss"]


def epoch_loss_metrics(task, train_loss, dev_loss, loss_ema, train_components, dev_components):
    """W&B epoch payload; non-S2A tasks keep their historical audio metric."""
    loss_name = "joint_loss" if task == "s2a" else "audio_loss"
    metrics = {
        f"train/{task}/{loss_name}": train_loss,
        f"train/{task}/{loss_name}_ema": loss_ema["joint_loss"],
        f"dev/{task}/{loss_name}": dev_loss,
    }
    if task == "s2a":
        for name in ("text_loss", "audio_loss"):
            metrics[f"train/{task}/{name}"] = train_components[name]
            metrics[f"train/{task}/{name}_ema"] = loss_ema[name]
            metrics[f"dev/{task}/{name}"] = dev_components[name]
    return metrics


def reject_audio_checkpoint(path):
    """This entry point starts from a text backbone, never silently resets audio."""
    path = Path(path)
    if (path / "mimi_audio_streams.pt").exists() or (path / "multitask_metadata.json").exists():
        raise ValueError("--qwen3-model is an audio checkpoint; use --init-checkpoint with a format-v5 Thinker-Talker checkpoint")
    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if any(key in config for key in ("multitask_format_version", "audio_stop_id", "audio_head_size", "audio_vocab_size")):
            raise ValueError("--qwen3-model is an audio checkpoint; use the original text backbone")
    keys = []
    for index in path.glob("*.index.json"):
        keys.extend(json.loads(index.read_text()).get("weight_map", {}))
    # Reading safetensors headers avoids materialising backbone weights.
    for weights in path.glob("*.safetensors"):
        with weights.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if size > 100_000_000:
                raise ValueError(f"invalid safetensors header: {weights}")
            keys.extend(json.loads(stream.read(size)))
    if not keys:
        for weights in path.glob("pytorch_model*.bin"):
            state = torch.load(weights, map_location="cpu", weights_only=True)
            keys.extend(state)
            del state
    if any("audio_streams." in key for key in keys):
        raise ValueError("--qwen3-model contains audio_streams weights; use --init-checkpoint with a format-v5 Thinker-Talker checkpoint")


def prepare_stage_output(run_root: Path, stage_dir: Path) -> None:
    """Create the run root, then reserve this stage's directory inside it.

    Every stage of one run shares ``run_root``, so earlier stages are expected to
    be there already; only ``stage_dir`` has to be new. Checked before any output
    writes, and rank-zero failures are shared with all ranks.
    """
    error = [None]
    if ddp_utils.is_main():
        try:
            if run_root.exists() and not run_root.is_dir():
                raise ValueError(f"--output is not a directory: {run_root}")
            run_root.mkdir(parents=True, exist_ok=True)
            if stage_dir.exists():
                raise ValueError(f"stage output already exists: {stage_dir}; pass a new --stage-index or --output")
            stage_dir.mkdir(parents=True)
        except (OSError, ValueError) as exc:
            error[0] = str(exc)
    if ddp_utils.is_distributed():
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])


def checkpoint_metadata(model=None):
    metadata = {
        "multitask_format_version": FORMAT_VERSION,
        "architecture": "thinker_talker",
        "audio_codebook_delays": list(AUDIO_CODEBOOK_DELAYS),
        "audio_sequence_layout": AUDIO_SEQUENCE_LAYOUT,
        "audio_stop_id": AUDIO_STOP_ID,
        "audio_vocab_size": AUDIO_OUTPUT_VOCAB_SIZE,
        "audio_head_size": AUDIO_HEAD_SIZE,
        "audio_input_vocab_size": AUDIO_INPUT_VOCAB_SIZE,
        "audio_pad_id": AUDIO_PAD_ID,
        "audio_speaker_id": AUDIO_SPEAKER_ID,
        "history_corruption": "shifted_supervised_times; all8_uniform2112; independent_text_uniform_vocab; protect_image_input; default_p0.05",
        "generation": GENERATION_CONFIG,
        "audio_codebook_size": CODEBOOK_SIZE,
        "num_codebooks": NUM_CODEBOOKS,
        "audio_stop_weight": AUDIO_STOP_WEIGHT,
        "loss_normalization": "text_mean_plus_sum_codebook_means_div_8; stop_weighted_numerator_only",
        "stage_schedule": "s2a_only; optional_format_v5_weight_warm_start; optimizer_scheduler_reset_per_run",
        "thinker_directory": "thinker",
        "talker_state": "talker.pt",
        "upstream_revision": "f900448c608318c53314ebf8a947ab05cd8c038e",
        "upstream_departures": {
            "codec_ids": "real codes 0..2047; STOP=2050 input/output; PAD=2049; vocab=2112",
            "speaker_conditioning": "unsupported; no speaker embeddings or speaker data",
        },
        "bridge_layer_convention": "zero_based_block_output_before_final_norm",
        "audio_embedding_aggregation": "mean_over_8_codebooks; inactive_uses_learned_PAD2049",
        "conditioning_scales": "learnable_text_and_audio; initialized_3_and_1",
    }
    if model is not None:
        metadata["talker_config"] = model.talker_config()
    return metadata


def parse_task(value: str) -> str:
    for task in TASKS:
        if value == task:
            return task
    raise argparse.ArgumentTypeError(f"this training entry supports only --task s2a, got {value!r}")


STAGE_DIR_PATTERN = re.compile(r"^stage_(\d+)_([a-z0-9_]+)$")


def existing_stages(run_root: Path) -> list[tuple[int, str]]:
    """``(index, task)`` of every stage already saved under ``run_root``."""
    if not run_root.is_dir():
        return []
    stages = []
    for child in run_root.iterdir():
        match = STAGE_DIR_PATTERN.match(child.name)
        if child.is_dir() and match:
            stages.append((int(match.group(1)), match.group(2)))
    return sorted(stages)


def next_stage_index(run_root: Path) -> int:
    """One past the highest stage index under ``run_root`` (``1`` when empty)."""
    return max((index for index, _ in existing_stages(run_root)), default=0) + 1


def stage_directory(run_root: Path, index: int, task: str) -> Path:
    if index < 1:
        raise ValueError("--stage-index must be positive")
    return run_root / f"stage_{index:02d}_{task}"


def validate_init_checkpoint(path):
    """Require the complete dual-decoder checkpoint before materialising weights."""
    path = Path(path)
    metadata_path = path / "multitask_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"incomplete audio checkpoint: {path}; metadata is required")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("multitask_format_version") != FORMAT_VERSION or metadata.get("architecture") != "thinker_talker":
        raise ValueError("only format-v5 thinker_talker checkpoints are supported; format-v4 and older weights are incompatible")
    settings = metadata.get("talker_config")
    if (not isinstance(settings, dict)
            or set(settings) != {"num_talker_layers", "bridge_layer", "adapter_rank"}
            or any(type(value) is not int for value in settings.values())
            or settings["num_talker_layers"] < 1 or settings["bridge_layer"] < 0
            or settings["adapter_rank"] < 1):
        raise ValueError("invalid format-v5 talker_config")
    if not (path / "thinker" / "config.json").is_file() or not (path / "talker.pt").is_file():
        raise ValueError("incomplete format-v5 checkpoint: separate thinker/ and talker.pt required")
    for key, expected in checkpoint_metadata().items():
        if key != "stage_schedule" and metadata.get(key) != expected:
            raise ValueError(f"incompatible audio checkpoint: {key}={metadata.get(key)!r}, expected {expected!r}")
    return metadata


def load_multitask_checkpoint(path, device):
    """Warm-start weights/tokenizer, not optimizer, schedule, epoch or RNG state."""
    path = Path(path)
    metadata = validate_init_checkpoint(path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    vocab = tokenizer.get_vocab()
    if any(token not in vocab for token in ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>")):
        raise ValueError("audio checkpoint tokenizer is missing audio markers")
    # Do not prepare_for_speech here: it can reinitialize trained marker rows.
    thinker, info = AutoModelForCausalLM.from_pretrained(
        str(path / "thinker"), local_files_only=True, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"incomplete or incompatible Thinker checkpoint: {info}")
    if max(vocab.values()) >= thinker.get_input_embeddings().num_embeddings:
        raise ValueError("checkpoint tokenizer exceeds Thinker embedding vocabulary")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3ThinkerTalker(thinker, **metadata["talker_config"], initialize_from_thinker=False)
    state = torch.load(path / "talker.pt", map_location="cpu", weights_only=True)
    model.audio_streams.load_state_dict(state, strict=True)
    return model.to(device).eval(), tokenizer


def save_checkpoint(base, tokenizer, path, task=None, epoch=0):
    path = Path(path)
    if path.exists():
        raise ValueError(f"checkpoint path already exists; refusing to overwrite: {path}")
    base.thinker.save_pretrained(path / "thinker")
    torch.save(base.audio_streams.state_dict(), path / "talker.pt")
    tokenizer.save_pretrained(path)
    (path / "multitask_metadata.json").write_text(
        json.dumps(checkpoint_metadata(base) | {"task": task, "epoch": epoch}
                   | ({"s2a_semantics": S2A_SEMANTICS} if task == "s2a" else {}), indent=2) + "\n"
    )


def load_stage_dataset(data, task, split, *, lang_filter=None, limit=0, seed=7,
                       tokenizer=None, max_seq_len=DEFAULT_S2A_MAX_SEQ_LEN):
    """S2A needs the tokenizer to render prompts; the dataset owns truncation/pad."""
    if Path(data).suffix.lower() == ".parquet":
        from dataset.minimind_t2a_dataset import MiniMindT2ADataset

        return MiniMindT2ADataset(data, task=task, split=split, lang_filter=lang_filter, limit=limit,
                                  seed=seed, tokenizer=tokenizer, max_seq_len=max_seq_len)
    if task == "s2a":
        raise ValueError("s2a requires MiniMind-O conversation Parquet; Emilia JSONL has no supported conversation contract")
    return EmiliaTaskDataset(Path(data) / f"{split}.jsonl", task, None,
                             lang_filter=lang_filter, limit=limit, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker-layers", type=int, default=4)
    parser.add_argument("--bridge-layer", type=int, default=None,
                        help="zero-based intermediate Thinker block; default middle block")
    parser.add_argument("--data", type=Path, required=True,
                        help="MiniMind-O conversation .parquet (first-user conversation split, 98/1/1)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qwen3-model", type=Path, help="fresh training from a base Qwen3 directory")
    source.add_argument("--init-checkpoint", type=Path, help="saved stage epoch directory; load weights/tokenizer only, start a new schedule")
    parser.add_argument("--output", type=Path, default=Path("outputs/05_route_b_multitask"),
                        help="run root; each command writes a new stage_{NN}_s2a/ directory inside it")
    parser.add_argument("--task", type=parse_task, choices=TASKS, default="s2a",
                        help="only S2A training is supported (default: s2a)")
    parser.add_argument("--stage-index", type=int, default=None,
                        help="stage number for the output directory prefix; defaults to one past the highest stage under --output")
    parser.add_argument("--epochs", type=int, default=3, help="full epochs for this stage")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5, help="Qwen backbone learning rate")
    parser.add_argument("--audio-lr", type=float, default=2e-4, help="Talker decoder, projections, codec embeddings and heads learning rate")
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--loss-ema", type=float, default=0.02)
    parser.add_argument("--max-frames", type=int, default=None, help="deprecated; ignored (full audio is preserved)")
    parser.add_argument("--max-length", type=int, default=None, help="deprecated; ignored (model context limit is used)")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_S2A_MAX_SEQ_LEN,
                        help="S2A conversation length: prompts longer after the backward fallback are truncated, then padded to exactly this width")
    parser.add_argument("--history-noise-prob", type=float, default=0.05,
                        help="training-only independent shifted-label text/audio corruption probability; audio replaces all 8 cells")
    parser.add_argument("--loss-chunk", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lang-filter", choices=("zh", "en"), default=None)
    parser.add_argument("--limit", type=int, default=0, help="maximum selected training samples; 0 uses all")
    parser.add_argument("--dev-limit", type=int, default=None, help="maximum dev samples; defaults to --limit, 0 uses all")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="Speech-MiniMind")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    if args.talker_layers < 1 or (args.bridge_layer is not None and args.bridge_layer < 0):
        parser.error("talker-layers must be positive and bridge-layer nonnegative")
    if not 0.0 <= args.history_noise_prob <= 1.0:
        parser.error("history-noise-prob must be in [0, 1]")
    if args.limit < 0 or (args.dev_limit is not None and args.dev_limit < 0):
        parser.error("limit and dev-limit must be nonnegative")
    if args.dev_limit is None:
        args.dev_limit = args.limit
    if args.data.suffix.lower() != ".parquet":
        parser.error("s2a requires conversation Parquet; Emilia JSONL conversations are unsupported")
    if args.grad_accum_steps < 1 or args.batch_size < 1:
        raise SystemExit("batch-size and grad-accum-steps must be positive")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    if args.max_frames is not None or args.max_length is not None:
        print("WARNING: --max-frames and --max-length are deprecated and ignored; "
              "full audio/text are preserved, subject only to the model context limit.", flush=True)
    args.max_frames = None
    if args.lr <= 0 or args.audio_lr <= 0:
        raise SystemExit("lr and audio-lr must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0 or not 0.0 <= args.min_lr_ratio <= 1.0 or not 0.0 <= args.loss_ema <= 1.0:
        raise SystemExit("warmup-ratio must be in [0,1), min-lr-ratio and loss-ema in [0,1]")
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    task = args.task
    stages = existing_stages(args.output)
    if args.stage_index is None:
        args.stage_index = next_stage_index(args.output)
    if args.stage_index < 1:
        raise SystemExit("--stage-index must be positive")
    stage_dir = stage_directory(args.output, args.stage_index, task)
    if any(index == args.stage_index for index, _ in stages):
        raise SystemExit(f"stage_{args.stage_index:02d}_* already exists under {args.output}; pass a new --stage-index")
    if args.init_checkpoint is not None:
        metadata = validate_init_checkpoint(args.init_checkpoint)
        settings = metadata["talker_config"]
        if settings["num_talker_layers"] != args.talker_layers or (
                args.bridge_layer is not None and settings["bridge_layer"] != args.bridge_layer):
            parser.error("requested Talker configuration does not match checkpoint")
    else:
        reject_audio_checkpoint(args.qwen3_model)
    torch.manual_seed(args.seed)
    ddp_utils.setup(); device = ddp_utils.device()
    if args.init_checkpoint is not None:
        model, tokenizer = load_multitask_checkpoint(args.init_checkpoint, device)
    else:
        model, tokenizer = load_qwen3(args.qwen3_model, device)
    prepare_stage_output(args.output, stage_dir)
    args.max_length = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(args.max_length, int) or args.max_length < 1:
        raise ValueError("model config must declare a positive max_position_embeddings context limit")
    if args.wandb and ddp_utils.is_main():
        if wandb is None:
            raise SystemExit("wandb not installed; install it in the speech-llm environment before using --wandb")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    register_audio_special_tokens(tokenizer)
    spec = build_vocab_spec(tokenizer, CODEBOOK_SIZE, NUM_CODEBOOKS)
    if args.init_checkpoint is None:
        model = Qwen3ThinkerTalker(model, num_talker_layers=args.talker_layers, bridge_layer=args.bridge_layer)
    for parameter in model.parameters(): parameter.requires_grad_(True)
    base = model
    model = MultitaskForward(base, args.loss_chunk, args.history_noise_prob,
                             text_vocab_size=len(tokenizer),
                             image_token_id=tokenizer.get_vocab().get("<|image_pad|>"))
    if ddp_utils.is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_utils.local_rank()] if device.type == "cuda" else None,
            output_device=ddp_utils.local_rank() if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    audio_parameters = list(base.audio_streams.parameters())
    audio_parameter_ids = {id(parameter) for parameter in audio_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in audio_parameter_ids
    ]
    train_set = load_stage_dataset(args.data, task, "train", lang_filter=args.lang_filter, limit=args.limit,
                                   seed=args.seed, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dev_set = load_stage_dataset(args.data, task, "dev", lang_filter=args.lang_filter, limit=args.dev_limit,
                                 seed=args.seed, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    train = make_loader(train_set, args.batch_size, device, True, ddp_utils.make_sampler(train_set, shuffle=True), args.num_workers)
    dev = make_loader(dev_set, args.batch_size, device, False, ddp_utils.make_sampler(dev_set, shuffle=False), args.num_workers)
    if len(train) == 0 or len(dev) == 0:
        raise ValueError(f"task {task} must have nonempty train and dev loaders")
    if ddp_utils.is_main():
        metadata = vars(args) | checkpoint_metadata(base)
        metadata["dataset"] = {
            "format": "minimind_o_t2a_parquet",
            "train_samples": len(train_set), "dev_samples": len(dev_set),
            "split_policy": getattr(train_set, "split_policy", "provided_train_dev_manifests"),
            "max_seq_len": getattr(train_set, "max_seq_len", None),
            "s2a_semantics": S2A_SEMANTICS,
        }
        (stage_dir / "config.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
        print(f"stage={args.stage_index:02d}_{task} train rows={len(train.dataset)} dev rows={len(dev.dataset)}; "
              f"init={args.init_checkpoint if args.init_checkpoint is not None else args.qwen3_model}; "
              f"epochs={args.epochs}; accumulation={args.grad_accum_steps}")
    # Only moments, schedule and logging EMA are fresh; weights came from --init-checkpoint.
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.lr},
        {"params": audio_parameters, "lr": args.audio_lr},
    ])
    updates_per_epoch = math.ceil(len(train) / args.grad_accum_steps)
    total_steps = args.epochs * updates_per_epoch
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = max(1, int(args.warmup_ratio * total_steps))
        floor = args.min_lr_ratio

        def lr_scale(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    loss_ema = None
    for epoch in range(1, args.epochs + 1):
        train_loss, loss_ema, train_components = run_epoch(
            args, model, tokenizer, spec, {task: train}, device, optimizer, scheduler,
            epoch, loss_ema, return_components=True)
        dev_loss, _, dev_components = run_epoch(
            args, model, tokenizer, spec, {task: dev}, device, epoch=epoch,
            return_components=True)
        if ddp_utils.is_main():
            save_checkpoint(base, tokenizer, stage_dir / f"model_epoch_{epoch:03d}", task, epoch)
            print(f"stage={task} epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")
            if args.wandb:
                wandb.log({
                    **epoch_loss_metrics(task, train_loss, dev_loss, loss_ema,
                                         train_components, dev_components),
                    "stage": args.stage_index,
                    "task": task,
                    "epoch": epoch,
                })
    if args.wandb and ddp_utils.is_main():
        wandb.finish()
    ddp_utils.cleanup()


if __name__ == "__main__":
    main()
