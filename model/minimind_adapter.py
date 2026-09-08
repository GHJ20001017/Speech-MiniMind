"""Helpers for using a MiniMind Transformers model with speech prefixes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def load_minimind(model_path: Path, device: torch.device):
    """Load a Transformers-format MiniMind checkpoint.

    The MiniMind repository is kept outside this project. Its custom model
    code is loaded through ``trust_remote_code`` so the user can update it
    independently of Speech-MiniMind.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use chapter 03: pip install transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer


def forward_inputs_embeds(model: nn.Module, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Run MiniMind's decoder when the prefix is continuous speech embeddings.

    MiniMind's native educational forward path starts from ``input_ids``.
    This small equivalent path preserves its rotary embeddings and decoder
    blocks while allowing a speech projector to provide the initial embeddings.
    """
    # Current minimind-3 is exported as a standard Qwen3ForCausalLM and
    # already supports inputs_embeds. Prefer that public interface.
    try:
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return output.logits
    except (TypeError, AttributeError):
        pass

    core = model.model
    batch_size, sequence_length, _ = inputs_embeds.shape
    hidden = core.dropout(inputs_embeds)
    if core.freqs_cos.device != hidden.device:
        core.freqs_cos = core.freqs_cos.to(hidden.device)
        core.freqs_sin = core.freqs_sin.to(hidden.device)
    position_embeddings = (
        core.freqs_cos[:sequence_length],
        core.freqs_sin[:sequence_length],
    )
    past = [None] * len(core.layers)
    for layer, past_key_value in zip(core.layers, past):
        hidden, _ = layer(
            hidden,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=False,
            attention_mask=attention_mask,
        )
    hidden = core.norm(hidden)
    return model.lm_head(hidden)


def token_embeddings(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    return model.model.embed_tokens(token_ids)
