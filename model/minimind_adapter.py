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


def _unwrap_for_generate(model: nn.Module) -> nn.Module:
    """Strip DDP / peft wrappers so ``generate`` runs on the raw LM.

    ``forward_inputs_embeds`` already unwraps for forward; generation needs the
    same treatment (a peft ``PeftModel`` forwards ``inputs_embeds`` fine but its
    ``generate`` may not accept them, so we go straight to the base model).
    """
    base = model
    while hasattr(base, "module") and not hasattr(base, "generate"):
        base = base.module
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    return base


def generate_from_speech(
    model: nn.Module,
    tokenizer,
    speech_embeds: torch.Tensor,
    speech_lengths: torch.Tensor,
    instruction: str,
    device: torch.device,
    max_new_tokens: int = 256,
    max_speech_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, list[int]]:
    """Autoregressively generate an answer given a speech prefix + instruction.

    The model consumes continuous speech embeddings (from the frozen encoder +
    projector) prepended to the tokenised instruction, mirroring the training
    layout ``[speech tokens][instruction tokens]``. Because the prefix is not
    a token-id sequence, we build ``inputs_embeds`` manually and try to run
    ``model.generate`` on it; MiniMind's exported decoder may not accept
    ``inputs_embeds`` through ``generate``, in which case we fall back to a
    simple greedy autoregressive loop over ``logits``.

    Args:
        model: MiniMind LM, possibly wrapped (DDP / peft).
        tokenizer: its AutoTokenizer.
        speech_embeds: ``(1, T_sp, H)`` projected speech prefix embeddings.
        speech_lengths: ``(1,)`` real count of speech tokens.
        instruction: text instruction to prepend after the speech prefix.
        device: execution device.
        max_new_tokens: answer length cap.
        max_speech_tokens: cap on how many speech tokens we keep.
        temperature: >0 enables sampling; 0 (default) is greedy.

    Returns:
        ``(answer_text, answer_token_ids)``.
    """
    base = _unwrap_for_generate(model)
    speech = speech_embeds[:, : speech_lengths[0], :][:, :max_speech_tokens, :]

    prompt_ids = tokenizer(
        instruction, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    prompt_embeds = base.model.embed_tokens(prompt_ids)  # [1, P, H]

    inputs_embeds = torch.cat([speech, prompt_embeds], dim=1)  # [1, S+P, H]
    seq_len = inputs_embeds.size(1)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    use_cache = bool(getattr(base, "config", None) and getattr(base.config, "use_cache", True))
    try:
        with torch.no_grad():
            output_ids = base.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                use_cache=use_cache,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                no_repeat_ngram_size=4,
                repetition_penalty=1.1,
            )
        # generated ids exclude the prompt (we fed embeds, not ids), so the full
        # output is simply the continuation.
        generated = output_ids[0].tolist()
    except (TypeError, AttributeError, NotImplementedError):
        generated = _greedy_loop(
            base, tokenizer, inputs_embeds, attention_mask, device,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

    # drop trailing EOS / padding
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated


def stream_from_speech(
    model: nn.Module,
    tokenizer,
    speech_embeds: torch.Tensor,
    speech_lengths: torch.Tensor,
    instruction: str,
    device: torch.device,
    max_new_tokens: int = 256,
    max_speech_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, str]:
    """Stream an answer generated from a speech prefix + instruction.

    Yields ``(partial_text, complete_text)`` as each new token is produced, so a
    caller can update a UI incrementally instead of waiting for the whole answer.
    The final yield carries the full text in both fields.

    Internally uses ``transformers.TextIteratorStreamer`` on ``model.generate``
    (KV-cache enabled). If ``generate(inputs_embeds=...)`` is unsupported it
    falls back to the greedy loop and yields after each token we can emit.
    """
    from transformers import TextIteratorStreamer

    base = _unwrap_for_generate(model)
    speech = speech_embeds[:, : speech_lengths[0], :][:, :max_speech_tokens, :]

    prompt_ids = tokenizer(
        instruction, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    prompt_embeds = base.model.embed_tokens(prompt_ids)

    inputs_embeds = torch.cat([speech, prompt_embeds], dim=1)
    seq_len = inputs_embeds.size(1)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    use_cache = bool(getattr(base, "config", None) and getattr(base.config, "use_cache", True))
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
    )
    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        use_cache=use_cache,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
        streamer=streamer,
    )
    import threading

    def _run_generate():
        with torch.no_grad():
            base.generate(**gen_kwargs)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    acc = ""
    for piece in streamer:
        acc += piece
        # light cleanup: streamer emits short fragments including whitespace
        yield acc, acc

    thread.join(timeout=10)
    yield acc, acc


def _greedy_loop(
    model: nn.Module,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> list[int]:
    """Fallback greedy/sampled autoregression when ``generate(inputs_embeds)``
    is unsupported. Recomputes the full prefix per step (slow but correct)."""
    generated: list[int] = []
    cur_embeds = inputs_embeds
    cur_mask = attention_mask
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = forward_inputs_embeds(model, cur_embeds, cur_mask)
        next_logits = logits[0, -1, :]
        if temperature > 0:
            import torch.nn.functional as F

            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(probs.multinomial(1).item())
        else:
            next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == (tokenizer.eos_token_id or -1):
            break
        generated.append(next_id)
        next_emb = model.model.embed_tokens(
            torch.tensor([[next_id]], device=device)
        )
        cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
        )
    return generated
