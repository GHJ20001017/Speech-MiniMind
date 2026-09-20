"""Helpers for running Qwen3-0.6B with continuous speech prefixes.

Qwen3-0.6B is a stock ``Qwen3ForCausalLM``: it accepts ``inputs_embeds`` on both
``forward`` and ``generate``, so splicing speech embeddings into the prompt needs
no custom decoder path. What this module adds is (a) loading the checkpoint in
the shape speech training expects - audio markers registered, embeddings given a
neutral starting value, the chat layout verified against the real tokenizer - and
(b) the "generate from speech" entry points shared by the CLI, the evaluator and
the web UI.

The speech embeddings reach the model as a continuous prefix inside Qwen3's
``user`` turn, laid out by :mod:`model.chat_format`::

    <|im_start|>system\\n{instruction}<|im_end|>\\n<|im_start|>user\\n<|audio_start|>
    [speech embeddings]
    <|audio_end|><|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>

Thinking stays disabled throughout: the empty think block is part of the prompt we
supervise, so the model answers immediately instead of opening a reasoning block.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch
from torch import nn

from model.chat_format import (
    encode_assistant_header,
    encode_prompt,
    prepare_llm_for_speech,
    verify_chat_layout,
)


def _resolve_dtype(dtype):
    """``None`` -> whatever the checkpoint says; else a torch dtype or its name."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype))


def load_qwen3(model_path: Path, device: torch.device, dtype=None,
               prepare_for_speech: bool = True):
    """Load a Qwen3-0.6B checkpoint for speech conditioning.

    ``prepare_for_speech`` registers the audio special tokens (see
    ``model.chat_format``) and verifies that the hand-built non-thinking layout
    still matches the tokenizer's own chat template. That check is the one thing
    that has to hold for training and inference to agree, so it runs on every
    load rather than only in tests.

    ``dtype`` accepts a torch dtype or its name. ``None`` (the default) leaves the
    choice to transformers, which loads float32 master weights even for a
    bfloat16 checkpoint - the safest thing to fine-tune, at twice the memory of
    the released weights. Pass ``"bfloat16"`` or ``"float16"`` to halve backbone
    memory when the weights stay frozen (projector / LoRA stages); the projected
    speech embeddings are cast to the backbone dtype at the handoff either way.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers to use the Qwen3 backbone: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if dtype is None:
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    else:
        resolved = _resolve_dtype(dtype)
        # `dtype` is the current name; older transformers only knows torch_dtype.
        try:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=resolved)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=resolved
            )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if prepare_for_speech:
        prepare_llm_for_speech(model, tokenizer)
        verify_chat_layout(tokenizer)
    return model.to(device).eval(), tokenizer


def forward_inputs_embeds(model: nn.Module, inputs_embeds: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    """Run the decoder over a prefix that mixes text and speech embeddings.

    Materialises the full ``(batch, length, vocab)`` logits, which is fine for
    generation and evaluation but is the dominant allocation while training a
    full fine-tune - the trainers therefore go through
    :func:`forward_hidden_states` and ``model/chunked_loss.py`` instead.
    """
    return model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits


def forward_hidden_states(model: nn.Module, attention_mask: torch.Tensor,
                          input_ids: torch.Tensor | None = None,
                          inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
    """Run the decoder stack only and return its final hidden states.

    Exactly one of ``input_ids`` / ``inputs_embeds`` is passed.  Skipping the LM
    head is what lets the callers chunk their cross-entropy, so the vocab-sized
    logits never have to exist in full (see ``model/chunked_loss.py``); the
    forward itself is the same one ``model(...)`` would run before the head.
    Works for a DDP-wrapped and/or peft-wrapped Qwen3 - the chat/lora adapters
    live inside these layers, not beside them.
    """
    core = _embedding_module(model)
    return core.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state


def lm_head_module(model: nn.Module) -> nn.Module:
    """The base LM's ``lm_head`` (DDP and peft unwrapped).

    Needed by the chunked loss, which applies the head itself. On Qwen3-0.6B
    ``lm_head`` is tied to ``embed_tokens``, so its gradients land in the same
    row-extended table the embeddings use.
    """
    return _embedding_module(model).lm_head


def token_embeddings(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """Look up embedding rows, unwrapping DDP / peft to reach ``embed_tokens``."""
    return _embedding_module(model).get_input_embeddings()(token_ids)


def _unwrap_ddp(model: nn.Module) -> nn.Module:
    """Strip a DDP wrapper, keeping peft adapters in the graph.

    Deliberately does *not* unwrap peft: ``PeftModel.generate`` runs the adapters,
    whereas the underlying base model would silently generate from the untuned
    weights.
    """
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base


def _embedding_module(model: nn.Module) -> nn.Module:
    """The module holding ``embed_tokens`` (DDP and peft both unwrapped)."""
    base = _unwrap_ddp(model)
    return base.get_base_model() if hasattr(base, "get_base_model") else base


def _sampling_kwargs(temperature: float, top_p: float, top_k: int) -> dict:
    """Qwen3's non-thinking guidance is greedy; sampling is opt-in via temperature."""
    if temperature and temperature > 0:
        kwargs = {"do_sample": True, "temperature": temperature}
        if top_p and top_p < 1.0:
            kwargs["top_p"] = top_p
        if top_k and top_k > 0:
            kwargs["top_k"] = top_k
        return kwargs
    return {"do_sample": False}


def _prompt_embeds(model, tokenizer, speech_embeds, speech_lengths, instruction,
                   device, max_speech_tokens):
    """Concatenate text-prefix + speech + assistant-header embeddings."""
    speech = speech_embeds[:, : speech_lengths[0], :][:, :max_speech_tokens, :]

    prefix_ids = torch.tensor(
        encode_prompt(tokenizer, instruction), dtype=torch.long, device=device
    ).unsqueeze(0)
    header_ids = torch.tensor(
        encode_assistant_header(tokenizer), dtype=torch.long, device=device
    ).unsqueeze(0)
    embed = _embedding_module(model)
    prefix_embeds = embed.get_input_embeddings()(prefix_ids)  # [1, P, H]
    header_embeds = embed.get_input_embeddings()(header_ids)  # [1, Q, H]

    # The projector runs in its own dtype; the decoder only accepts one.
    speech = speech.to(prefix_embeds.dtype)
    inputs_embeds = torch.cat([prefix_embeds, speech, header_embeds], dim=1)
    attention_mask = torch.ones(
        (1, inputs_embeds.size(1)), dtype=torch.long, device=device
    )
    return inputs_embeds, attention_mask


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
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> tuple[str, list[int]]:
    """Generate an answer for a speech prefix plus instruction.

    Args:
        model: Qwen3-0.6B, possibly wrapped (DDP / peft).
        tokenizer: its tokenizer.
        speech_embeds: ``(1, T_sp, H)`` projected speech prefix embeddings.
        speech_lengths: ``(1,)`` real count of speech tokens.
        instruction: system prompt rendered into the ``system`` turn.
        device: execution device.
        max_new_tokens: answer length cap.
        max_speech_tokens: cap on how many speech tokens we keep.
        temperature: ``>0`` enables sampling; ``0`` (default) is greedy, which is
            what Qwen3 recommends for non-thinking mode.
        top_p / top_k: sampling filters, used only when ``temperature > 0``.
        repetition_penalty / no_repeat_ngram_size: disabled by default (``1.0`` /
            ``0``); Qwen3 does not need them and they distort the distribution.

    Returns:
        ``(answer_text, answer_token_ids)``.
    """
    gen = _unwrap_ddp(model)
    inputs_embeds, attention_mask = _prompt_embeds(
        model, tokenizer, speech_embeds, speech_lengths, instruction, device,
        max_speech_tokens,
    )
    use_cache = bool(getattr(getattr(gen, "config", None), "use_cache", True))
    try:
        with torch.no_grad():
            output_ids = gen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **_sampling_kwargs(temperature, top_p, top_k),
            )
        # We fed embeddings rather than ids, so the output is the continuation
        # alone - there is no prompt to slice off.
        generated = output_ids[0].tolist()
    except (TypeError, AttributeError, NotImplementedError):
        generated = _greedy_loop(
            gen, tokenizer, inputs_embeds, attention_mask, device,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

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
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    """Yield ``(partial_text, complete_text)`` as the answer is generated.

    Lets a UI render incrementally instead of waiting for the whole answer. Uses
    the same layout and decoding settings as :func:`generate_from_speech`, and
    falls back to the greedy loop when ``generate(inputs_embeds=...)`` is
    unsupported.
    """
    from transformers import TextIteratorStreamer

    gen = _unwrap_ddp(model)
    inputs_embeds, attention_mask = _prompt_embeds(
        model, tokenizer, speech_embeds, speech_lengths, instruction, device,
        max_speech_tokens,
    )
    use_cache = bool(getattr(getattr(gen, "config", None), "use_cache", True))
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
    )
    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streamer=streamer,
        **_sampling_kwargs(temperature, top_p, top_k),
    )

    def _run_generate():
        with torch.no_grad():
            gen.generate(**gen_kwargs)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    accumulated = ""
    for piece in streamer:
        accumulated += piece
        yield accumulated, accumulated
    thread.join(timeout=10)
    yield accumulated, accumulated


def _greedy_loop(
    model: nn.Module,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> list[int]:
    """Fallback autoregression when ``generate(inputs_embeds)`` is unsupported.

    Re-runs the whole prefix each step, so it is only a safety net - it is far
    slower than ``generate`` with a KV cache.
    """
    generated: list[int] = []
    cur_embeds = inputs_embeds
    cur_mask = attention_mask
    embed = _embedding_module(model)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = forward_inputs_embeds(model, cur_embeds, cur_mask)
        next_logits = logits[0, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(probs.multinomial(1).item())
        else:
            next_id = int(next_logits.argmax(dim=-1).item())
        if next_id == (tokenizer.eos_token_id or -1):
            break
        generated.append(next_id)
        next_emb = embed.get_input_embeddings()(
            torch.tensor([[next_id]], device=device)
        )
        cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
        )
    return generated
