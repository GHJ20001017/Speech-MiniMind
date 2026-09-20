"""Audio-only LLM plumbing for Route B (discrete codebook, end-to-end).

Route B models speech as a **flat sequence of discrete codebook tokens** using
the same Qwen3-0.6B decoder as Route A, so the LM needs three additions:

1. an **extended vocabulary** - text tokens stay, audio tokens occupy a
   contiguous block above the text vocab;
2. **special tokens** delimiting the input audio and the generated audio;
3. a **sequence layout + label mask** that trains only on the *generated*
   audio (the prompt audio and the prompt itself are ``-100``).

Layout for one Speech-to-Speech sample (``Q`` codebooks, ``M`` prompt frames,
``K`` answer frames)::

    [BOS] <|audio_start|>  prompt frames (M*Q tokens)  <|audio_end|>
          <|audio_start|>  answer frames (K*Q tokens)  <|audio_end|>  [EOS]
    labels = -100 everywhere except the answer frame tokens, their <|audio_end|>,
             and the closing [EOS]

Frames are **flattened codebook-major-then-frame** (``t*Q + q``), matching the
public MiniMind-O ``sft_a2a`` tokenisation.  ``Q == 1`` degenerates to a plain
single-codebook audio LM, which is why the same code path serves both the
single-codebook plan and the 8-codebook Mimi default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from model.chat_format import (
    AUDIO_END as AUDIO_EOS,
    AUDIO_PAD,
    AUDIO_SPECIAL_TOKENS,
    AUDIO_START as AUDIO_BOS,
    ensure_audio_tokens,
)

# The marker strings are defined once in :mod:`model.chat_format` and re-exported
# here under the Route-B names downstream call sites already import.  Qwen3-0.6B's
# released tokenizer does *not* define them (it spells ``<|audio_start|>`` out as
# six unrelated byte tokens), so :func:`register_audio_special_tokens` appends
# them above the text vocab - at ids 151669/151670/151671 on Qwen3-0.6B, still
# inside the 151936 embedding rows.  Reusing MiniMind-O's names keeps our audio
# codes readable against its public ``sft_a2a`` set.

@dataclass
class AudioVocabSpec:
    """Where the audio token block lives inside the LM vocabulary.

    ``audio_offset`` is the first audio-token id; codebook ``q`` owns the slice
    ``[audio_offset + q*codebook_size, audio_offset + (q+1)*codebook_size)``.
    On Qwen3-0.6B the three audio special tokens are *appended* to the released
    text vocab (ids 151669/151670/151671), so they sit inside the text block,
    below ``audio_offset = len(tokenizer) = 151672``, and the audio block starts
    right above them.
    """

    text_vocab_size: int
    codebook_size: int
    num_codebooks: int
    audio_offset: int = 0
    audio_bos_id: int = 0
    audio_eos_id: int = 0
    audio_pad_id: int = 0

    @property
    def audio_vocab_size(self) -> int:
        return self.codebook_size * self.num_codebooks

    @property
    def total_vocab_size(self) -> int:
        """Embedding rows needed: the audio block, and at least the special tokens.

        On Qwen3-0.6B the markers are appended to the text vocab, so they are
        already covered by ``audio_offset`` and the audio block alone decides the
        row count; the ``max`` keeps the invariant true for any layout where a
        marker id could land after the block.
        """
        special_span = max(self.audio_bos_id, self.audio_eos_id, self.audio_pad_id) + 1
        return max(self.audio_offset + self.audio_vocab_size, special_span)

    def token_id(self, codebook: int, code: int) -> int:
        """Map ``(codebook, code)`` to a single LM token id."""
        if not 0 <= codebook < self.num_codebooks:
            raise ValueError(f"codebook {codebook} out of range 0..{self.num_codebooks - 1}")
        if not 0 <= code < self.codebook_size:
            raise ValueError(f"code {code} out of range 0..{self.codebook_size - 1}")
        return self.audio_offset + codebook * self.codebook_size + code

    def decode_token_id(self, token_id: int) -> tuple[int, int]:
        relative = token_id - self.audio_offset
        if not 0 <= relative < self.audio_vocab_size:
            raise ValueError(f"token id {token_id} is not an audio token")
        return divmod(relative, self.codebook_size)

    def is_audio_token(self, token_id: int) -> bool:
        return self.audio_offset <= token_id < self.audio_offset + self.audio_vocab_size


def register_audio_special_tokens(tokenizer) -> None:
    """Make sure the three audio special tokens exist in the tokenizer.

    Qwen3-0.6B's released tokenizer does *not* ship them, so this appends
    ``<|audio_start|>`` / ``<|audio_end|>`` / ``<|audio_pad|>`` above the text
    vocab - on Qwen3-0.6B at ids 151669/151670/151671, still inside the 151936
    embedding rows, so the markers alone never force a resize.  A tokenizer saved
    by one of our own fine-tuned checkpoints already carries them, in which case
    nothing is appended.  Delegates to :func:`model.chat_format.ensure_audio_tokens`,
    the single definition of the markers shared with Route A.
    """
    ensure_audio_tokens(tokenizer)


def build_vocab_spec(
    tokenizer,
    codebook_size: int,
    num_codebooks: int,
) -> AudioVocabSpec:
    """Reserve the audio block directly above the tokenizer's text vocabulary.

    Registers the markers through :func:`model.chat_format.ensure_audio_tokens`
    (the same path Route A uses), so ``len(tokenizer)`` is the final text size -
    151672 on Qwen3-0.6B - and the audio block starts right above it.
    """
    ids, _ = ensure_audio_tokens(tokenizer)
    text_vocab_size = len(tokenizer)
    return AudioVocabSpec(
        text_vocab_size=text_vocab_size,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        audio_offset=text_vocab_size,
        audio_bos_id=ids[AUDIO_BOS],
        audio_eos_id=ids[AUDIO_EOS],
        audio_pad_id=ids[AUDIO_PAD],
    )


def extend_model_vocab(
    model,
    tokenizer,
    spec: AudioVocabSpec,
    init_std: float = 0.02,
) -> AudioVocabSpec:
    """Resize the LM embeddings/head to fit the audio block and init new rows.

    Text rows keep their pre-trained values; only the newly added rows are
    initialised, with a small normal distribution so the first forward pass does
    not explode.  On Qwen3-0.6B the audio block starts at ``len(tokenizer)`` =
    151672, which is *below* the released table's 151936 rows, so its first
    ``151936 - 151672`` rows keep the checkpoint's unused padding values and only
    rows ``[151936:]`` get the fresh init.  Returns the (possibly refreshed) spec.
    """
    spec = AudioVocabSpec(
        text_vocab_size=spec.text_vocab_size,
        codebook_size=spec.codebook_size,
        num_codebooks=spec.num_codebooks,
        audio_offset=spec.audio_offset,
        audio_bos_id=spec.audio_bos_id,
        audio_eos_id=spec.audio_eos_id,
        audio_pad_id=spec.audio_pad_id,
    )

    # Prefer `resize_token_embeddings` (also resizes the tied lm_head when tied).
    old_embeddings = model.get_input_embeddings().weight.data
    old_size = old_embeddings.size(0)
    target = max(spec.total_vocab_size, old_size)
    model.resize_token_embeddings(target)

    new_embeddings = model.get_input_embeddings().weight.data
    if new_embeddings.size(0) > old_size:
        new_rows = new_embeddings[old_size:]
        new_rows.normal_(mean=0.0, std=init_std)
        new_embeddings[:old_size].copy_(old_embeddings)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and not getattr(model.config, "tie_word_embeddings", False):
        weight = output_embeddings.weight.data
        if weight.size(0) > old_size:
            weight[old_size:].normal_(mean=0.0, std=init_std)

    model.config.vocab_size = new_embeddings.size(0)
    return spec


def flatten_frames(codes: torch.Tensor) -> torch.Tensor:
    """``(Q, T)`` -> ``(T*Q,)`` frame-major then codebook (``t*Q + q``)."""
    if codes.dim() != 2:
        raise ValueError(f"expected (num_codebooks, frames), got {tuple(codes.shape)}")
    return codes.transpose(0, 1).reshape(-1).contiguous()


def unflatten_tokens(tokens: torch.Tensor, num_codebooks: int) -> torch.Tensor:
    """Inverse of :func:`flatten_frames` for a *flat* code token sequence."""
    if tokens.numel() % num_codebooks:
        raise ValueError(
            f"token count {tokens.numel()} is not divisible by num_codebooks={num_codebooks}"
        )
    frames = tokens.view(-1, num_codebooks)
    return frames.transpose(0, 1).contiguous()


def to_lm_tokens(codes: torch.Tensor, spec: AudioVocabSpec) -> torch.Tensor:
    """Convert ``(Q, T)`` code ids to LM token ids ``(T*Q,)``."""
    flat = flatten_frames(codes)
    offsets = torch.arange(spec.num_codebooks, device=flat.device) * spec.codebook_size
    # frame-major layout: codebook index cycles fastest, so tile (not repeat_interleave).
    offsets = offsets.repeat(codes.size(1))
    return flat + spec.audio_offset + offsets


def from_lm_tokens(tokens: torch.Tensor, spec: AudioVocabSpec) -> torch.Tensor:
    """Inverse of :func:`to_lm_tokens` -> ``(Q, T)`` code ids."""
    relative = tokens - spec.audio_offset
    if relative.min().item() < 0 or relative.max().item() >= spec.audio_vocab_size:
        raise ValueError("token sequence contains non-audio ids")
    offsets = torch.arange(spec.num_codebooks, device=tokens.device) * spec.codebook_size
    offsets = offsets.repeat(tokens.numel() // spec.num_codebooks)
    return unflatten_tokens(relative - offsets, spec.num_codebooks)


@dataclass
class AudioSample:
    """One training sample in raw code space."""

    prompt_codes: torch.Tensor | None  # (Q, M) int64 or None for pure audio LM
    answer_codes: torch.Tensor          # (Q, K) int64
    prompt_text_ids: list[int] = field(default_factory=list)


def build_audio_batch(
    tokenizer,
    spec: AudioVocabSpec,
    samples: list[AudioSample],
    max_length: int = 1024,
    max_answer_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble a padded training batch.

    Returns ``(input_ids, labels, attention_mask)`` with shape ``(B, L)``.
    Only answer-audio tokens (plus their ``<|audio_end|>`` and the closing
    ``[EOS]``, so the model also learns to stop) are supervised.
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for sample in samples:
        answer_codes = sample.answer_codes
        if max_answer_frames and answer_codes.size(1) > max_answer_frames:
            answer_codes = answer_codes[:, :max_answer_frames]

        row: list[int] = []
        labels: list[int] = []
        if bos_id is not None:
            row.append(bos_id)
            labels.append(-100)
        row.extend(sample.prompt_text_ids)
        labels.extend([-100] * len(sample.prompt_text_ids))

        if sample.prompt_codes is not None and sample.prompt_codes.size(1) > 0:
            row.append(spec.audio_bos_id)
            labels.append(-100)
            prompt_tokens = to_lm_tokens(sample.prompt_codes, spec).tolist()
            row.extend(prompt_tokens)
            labels.extend([-100] * len(prompt_tokens))
            row.append(spec.audio_eos_id)
            labels.append(-100)

        answer_tokens = to_lm_tokens(answer_codes, spec).tolist()
        row.append(spec.audio_bos_id)
        labels.append(-100)
        row.extend(answer_tokens)
        labels.extend(answer_tokens)
        row.append(spec.audio_eos_id)
        labels.append(spec.audio_eos_id)
        if eos_id is not None:
            row.append(eos_id)
            labels.append(eos_id)

        row = row[:max_length]
        labels = labels[:max_length]
        input_rows.append(row)
        label_rows.append(labels)

    length = max(len(r) for r in input_rows)
    input_ids = torch.full((len(input_rows), length), pad_id, dtype=torch.long)
    labels = torch.full((len(input_rows), length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), length), dtype=torch.long)
    for index, (row, label) in enumerate(zip(input_rows, label_rows)):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        labels[index, : len(label)] = torch.tensor(label, dtype=torch.long)
        attention_mask[index, : len(row)] = 1
    return input_ids, labels, attention_mask


@torch.no_grad()
def generate_audio_tokens(
    model,
    spec: AudioVocabSpec,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_k: int = 0,
    eos_token_id: int | None = None,
) -> list[int]:
    """Autoregressively sample audio tokens after ``prompt_ids``.

    ``prompt_ids`` must already contain the ``<|audio_start|>`` that opens the
    answer. Returns the generated *audio* token ids (special tokens excluded).

    The frame-major layout fixes which codebook each position carries: slot ``n``
    of the answer belongs to codebook ``n % Q``.  Only that block (plus the audio
    end token) is kept in the distribution, so a partially trained model cannot
    drift into another codebook and emit ids that :func:`from_lm_tokens` maps
    back to an out-of-range code value.
    """
    device = next(model.parameters()).device
    generated: list[int] = []
    ids = prompt_ids.to(device)
    stop_id = eos_token_id if eos_token_id is not None else spec.audio_eos_id
    for step in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]
        next_logits = logits[0]
        # Restrict to the codebook that this slot is supposed to carry.
        codebook = step % spec.num_codebooks
        lower = spec.audio_offset + codebook * spec.codebook_size
        upper = lower + spec.codebook_size
        mask = torch.full_like(next_logits, float("-inf"))
        mask[lower:upper] = 0.0
        mask[stop_id] = 0.0
        next_logits = next_logits + mask
        if temperature and temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            if top_k:
                values, indices = torch.topk(probs, min(top_k, probs.numel()))
                pick = torch.multinomial(values / values.sum(), 1)
                next_id = int(indices[pick])
            else:
                next_id = int(torch.multinomial(probs, 1))
        else:
            next_id = int(next_logits.argmax(dim=-1))
        if next_id == stop_id:
            break
        if not spec.is_audio_token(next_id):
            # A text token in the audio stream means the model lost the format;
            # stop rather than emit garbage frames.
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return generated
