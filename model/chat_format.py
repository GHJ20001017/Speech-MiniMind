"""MiniMind-native chat-template layout for speech-conditioned SFT.

Training and inference must build the byte-identical sequence, otherwise the
fine-tuned model is asked to continue a prefix it never saw. The layout follows
MiniMind's own chat template, with the spoken utterance occupying the ``user``
turn::

    <|im_start|>system\\n{prompt}<|im_end|>\\n<|im_start|>user\\n<|audio_start|>{语音}<|audio_end|><|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>

Because ``{语音}`` is continuous speech embeddings rather than token ids, the
text is split into two segments around them:

* :func:`encode_prompt` - the system turn, the ``user`` turn opener and the
  opening audio marker, i.e. everything up to and including ``<|audio_start|>``
* :func:`encode_assistant_header` - the closing audio marker, the ``user`` turn
  close and the assistant turn opener

``<|audio_start|>`` / ``<|audio_end|>`` are reserved by MiniMind-3's tokenizer
(ids 14 / 15) and are the same pair ``model/audio_lm.py`` uses for route B, so
both routes signal "audio here" identically. The exported ``minimind-3``
checkpoint is a plain ``Qwen3ForCausalLM`` with no audio wiring of its own:
these are ordinary embedding rows, and the code below is what actually splices
the speech vectors between them.

Only the trailing ``{answer}<|im_end|>`` is supervised; both segments and the
speech embeddings are masked with ``-100``. Masking the assistant header (and
not only the answer body) mirrors MiniMind's ``SFTDataset.generate_labels``,
which starts the loss right after ``<|im_start|>assistant\\n``.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Reserved by MiniMind-3's own tokenizer (ids 14 / 15); the same strings are
# aliased as AUDIO_BOS / AUDIO_EOS in model/audio_lm.py for route B.
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"


def chat_prompt(system_prompt: str) -> str:
    """The prompt text that precedes the speech embeddings."""
    return f"{IM_START}system\n{system_prompt}{IM_END}\n{IM_START}user\n{AUDIO_START}"


def answer_text(tokenizer, answer: str) -> str:
    """``answer`` closed by the end-of-turn token, i.e. the supervised span."""
    return answer + (tokenizer.eos_token or IM_END)


def encode_prompt(tokenizer, system_prompt: str) -> list[int]:
    """Token ids before the speech embeddings (system turn, ``user`` opener, ``<|audio_start|>``)."""
    return tokenizer(chat_prompt(system_prompt), add_special_tokens=False)["input_ids"]


def encode_assistant_header(tokenizer) -> list[int]:
    """Token ids between the speech embeddings and the answer text."""
    return tokenizer(
        f"{AUDIO_END}{IM_END}\n{IM_START}assistant\n", add_special_tokens=False
    )["input_ids"]


def encode_answer(tokenizer, answer: str) -> list[int]:
    """Token ids of the supervised answer span (``{answer}<|im_end|>``)."""
    return tokenizer(answer_text(tokenizer, answer), add_special_tokens=False)["input_ids"]
