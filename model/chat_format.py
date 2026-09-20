"""Qwen3-0.6B (non-thinking) chat layout for speech-conditioned SFT.

Training and inference must build the token-identical sequence below, otherwise
the fine-tuned model is asked to continue a prefix it never saw. The layout is
Qwen3's own chat template with the spoken utterance placed inside the ``user``
turn::

    <|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|>{speech}<|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{answer}<|im_end|>

Thinking is disabled. In Qwen3 that means the template pre-fills an *empty*
think block after ``assistant\\n`` (``enable_thinking=False``), so the model
answers immediately instead of opening a reasoning block. Non-thinking is the
mode every stage of this project trains and evaluates in.

Because ``{speech}`` is continuous speech embeddings rather than token ids, the
text is split into two segments around them (see :func:`encode_prompt` and
:func:`encode_assistant_header`) and each segment is tokenised separately. Qwen3
uses BPE, so a split point could in principle merge differently than the whole
rendered string; :func:`verify_chat_layout` checks the *ids*, not just the text,
and :func:`load_qwen3` runs it once per load. A tokenizer update therefore fails
loudly instead of silently desynchronising training from inference.

Audio markers
-------------
``<|audio_start|>`` / ``<|audio_end|>`` are **not** in Qwen3's vocabulary: the
released tokenizer spells ``<|audio_start|>`` out as six unrelated byte tokens
(``[27, 91, 16736, 4906, 91, 29]``), which is why :func:`ensure_audio_tokens`
registers them as added special tokens and :func:`encode_prompt` refuses to run
before that happened. On Qwen3-0.6B they land at ids 151669/151670/151671, i.e.
inside the embedding table's 151936 rows, so no resize is needed - the rows
simply move from "unused padding" to "audio markers" and
:func:`prepare_llm_for_speech` gives them a neutral value. Route B reuses the
same pair for its discrete audio stream via ``model/audio_lm.py``, so both
routes signal "audio lives here" with the same tokens.

Only the trailing ``{answer}<|im_end|>`` is supervised; both text segments, the
speech embeddings and the empty think block are masked with ``-100``. Masking
the assistant header (and not only the answer body) follows Qwen3's own SFT
convention of starting the loss right after ``<|im_start|>assistant\\n``.

Deliberate deviation from the template
--------------------------------------
Qwen3's template ends a *completed* turn with ``{answer}<|im_end|>\\n``. This
module stops at ``<|im_end|>`` and drops the trailing newline, because the newline
is only a separator for whatever turn comes next and our sequences always end
there. Including it would put the newline inside the supervised span and teach
the model to emit one after every end-of-turn marker.
"""

from __future__ import annotations

import torch

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Registered as added special tokens; see ensure_audio_tokens.
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_SPECIAL_TOKENS = (AUDIO_START, AUDIO_END, AUDIO_PAD)

# Qwen3 renders exactly this after "assistant\n" when thinking is disabled: an
# empty reasoning block that is already closed, so the model answers at once.
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def prompt_text(system_prompt: str) -> str:
    """Everything before the speech embeddings, closing on ``<|audio_start|>``."""
    return f"{IM_START}system\n{system_prompt}{IM_END}\n{IM_START}user\n{AUDIO_START}"


def header_text(non_thinking: bool = True) -> str:
    """Everything between the speech embeddings and the answer body."""
    text = f"{AUDIO_END}{IM_END}\n{IM_START}assistant\n"
    return text + NON_THINKING_PREFIX if non_thinking else text


def answer_text(answer: str) -> str:
    """The supervised span: ``{answer}`` closed by the end-of-turn marker."""
    return answer + IM_END


def _require_audio_tokens(tokenizer) -> None:
    """Fail loudly rather than tokenising a marker into six byte tokens."""
    vocab = tokenizer.get_vocab()
    missing = [token for token in (AUDIO_START, AUDIO_END) if token not in vocab]
    if missing:
        raise RuntimeError(
            f"audio special tokens not registered: {missing}; "
            "call ensure_audio_tokens(tokenizer) or prepare_llm_for_speech(model, "
            "tokenizer) before encoding a speech layout"
        )


def encode_prompt(tokenizer, system_prompt: str) -> list[int]:
    """Token ids before the speech embeddings, including ``<|audio_start|>``."""
    _require_audio_tokens(tokenizer)
    return list(tokenizer(prompt_text(system_prompt), add_special_tokens=False)["input_ids"])


def encode_assistant_header(tokenizer, non_thinking: bool = True) -> list[int]:
    """Token ids between the speech embeddings and the answer body."""
    _require_audio_tokens(tokenizer)
    return list(
        tokenizer(header_text(non_thinking), add_special_tokens=False)["input_ids"]
    )


def encode_answer(tokenizer, answer: str) -> list[int]:
    """Token ids of the supervised ``{answer}<|im_end|>`` span."""
    return list(tokenizer(answer_text(answer), add_special_tokens=False)["input_ids"])


def audio_token_ids(tokenizer) -> dict[str, int]:
    """Current ids of the audio special tokens (must already be registered)."""
    _require_audio_tokens(tokenizer)
    return {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in AUDIO_SPECIAL_TOKENS
    }


def ensure_audio_tokens(tokenizer) -> tuple[dict[str, int], list[str]]:
    """Register the audio special tokens, appending only the ones missing.

    Qwen3's released tokenizer does not define them, so this is normally what
    introduces ``<|audio_start|>`` / ``<|audio_end|>`` / ``<|audio_pad|>``. A
    tokenizer saved by one of our own fine-tuned checkpoints already carries
    them, in which case nothing is appended.

    Returns each token's id plus the names just added, so the caller only
    initialises genuinely new embedding rows and never clobbers trained ones.
    """
    vocab = tokenizer.get_vocab()
    added = [token for token in AUDIO_SPECIAL_TOKENS if token not in vocab]
    if added:
        tokenizer.add_special_tokens({"additional_special_tokens": added})
    return audio_token_ids(tokenizer), added


def prepare_llm_for_speech(model, tokenizer, init: str = "mean") -> dict[str, int]:
    """Make ``model`` able to consume the audio special tokens.

    Registers the tokens, widens the embedding table only if their ids fall
    outside it, and gives any *newly added* rows a neutral starting value:

    * ``mean`` - mean of the tokenizer's real rows, in-distribution and
      non-degenerate (the default);
    * ``zero`` - an explicit "no signal" vector.

    Rows are left untouched when the checkpoint already carried the tokens, so
    re-loading a fine-tuned model never overwrites embeddings it learned.

    Note under ``--tune lora``: LoRA adapters do not cover ``embed_tokens``, so
    the marker rows keep this initial value for the whole run. That is fine for
    a boundary marker, but it does mean the markers carry no learned meaning -
    ``--tune full`` trains them along with everything else.
    """
    ids, added = ensure_audio_tokens(tokenizer)

    embeddings = model.get_input_embeddings()
    needed = max(ids.values()) + 1
    if needed > embeddings.weight.size(0):
        model.resize_token_embeddings(needed)
        embeddings = model.get_input_embeddings()

    if added:
        weight = embeddings.weight
        # Added tokens are appended, so the lowest new id is the first row that
        # was not part of the original vocabulary.
        trained_rows = min(ids[token] for token in added)
        with torch.no_grad():
            if init == "mean":
                fill = weight[:trained_rows].mean(dim=0)
            elif init == "zero":
                fill = torch.zeros_like(weight[0])
            else:
                raise ValueError(f"unknown init {init!r}; expected 'mean' or 'zero'")
            for token in added:
                weight[ids[token]] = fill

    return ids


def verify_chat_layout(
    tokenizer,
    system_prompt: str = "系统提示",
    answer: str = "回答",
) -> None:
    """Assert the hand-built layout equals Qwen3's own chat template.

    Compares the rendered *text* and the resulting *token ids*, because the two
    text segments are tokenised separately around the speech embeddings and a
    BPE merge across that split would not show up in a text-only comparison.
    The template's trailing newline is asserted to be exactly the one this
    module deliberately omits (see the module docstring).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AUDIO_START + AUDIO_END},
        {"role": "assistant", "content": answer},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        raise RuntimeError(
            "this tokenizer has no enable_thinking switch, so it is not a Qwen3 "
            "chat template; the non-thinking layout cannot be verified"
        )

    manual_text = prompt_text(system_prompt) + header_text() + answer_text(answer)
    if rendered != manual_text + "\n":
        raise RuntimeError(
            "chat layout drift: the hand-built non-thinking layout no longer "
            "matches tokenizer.apply_chat_template(enable_thinking=False)\n"
            f"  template: {rendered!r}\n  expected: {(manual_text + chr(10))!r}"
        )

    template_ids = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    manual_ids = (
        encode_prompt(tokenizer, system_prompt)
        + encode_assistant_header(tokenizer)
        + encode_answer(tokenizer, answer)
    )
    trailing = template_ids[len(manual_ids):]
    if template_ids[: len(manual_ids)] != manual_ids or len(trailing) != 1:
        raise RuntimeError(
            "chat layout tokenisation drift: splitting the layout around the "
            "speech embeddings no longer reproduces the template's token ids\n"
            f"  template: {template_ids}\n  expected: {manual_ids} + one newline"
        )
