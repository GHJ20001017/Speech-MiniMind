"""MiniMind-native chat-template layout for speech-conditioned SFT.

Training and inference must build the byte-identical sequence, otherwise the
fine-tuned model is asked to continue a prefix it never saw. The layout follows
MiniMind's own chat template, with the spoken utterance occupying the ``user``
turn::

    <|im_start|>system\\n{prompt}<|im_end|>\\n<|im_start|>user\\n{语音}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>

Because ``{语音}`` is continuous speech embeddings rather than token ids, the
text is split into two segments around them:

* :func:`encode_prompt` - the system turn plus the ``user`` turn opener, i.e.
  everything up to and including ``<|im_start|>user\\n``
* :func:`encode_assistant_header` - the ``user`` turn close plus the assistant
  turn opener

Only the trailing ``{answer}<|im_end|>`` is supervised; both segments and the
speech embeddings are masked with ``-100``. Masking the assistant header (and
not only the answer body) mirrors MiniMind's ``SFTDataset.generate_labels``,
which starts the loss right after ``<|im_start|>assistant\\n``.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def chat_prompt(system_prompt: str) -> str:
    """The full prompt text of a speech turn, before tokenisation."""
    return f"{IM_START}system\n{system_prompt}{IM_END}\n{IM_START}user\n"


def answer_text(tokenizer, answer: str) -> str:
    """``answer`` closed by the end-of-turn token, i.e. the supervised span."""
    return answer + (tokenizer.eos_token or IM_END)


def encode_prompt(tokenizer, system_prompt: str) -> list[int]:
    """Token ids before the speech embeddings (system turn + ``user`` opener)."""
    return tokenizer(chat_prompt(system_prompt), add_special_tokens=False)["input_ids"]


def encode_assistant_header(tokenizer) -> list[int]:
    """Token ids between the speech embeddings and the answer text."""
    return tokenizer(
        f"{IM_END}\n{IM_START}assistant\n", add_special_tokens=False
    )["input_ids"]


def encode_answer(tokenizer, answer: str) -> list[int]:
    """Token ids of the supervised answer span (``{answer}<|im_end|>``)."""
    return tokenizer(answer_text(tokenizer, answer), add_special_tokens=False)["input_ids"]
