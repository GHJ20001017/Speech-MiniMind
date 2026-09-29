"""Direct TTS and conversation-prefix S2A views of MiniMind frame-major audio.

TTS uses only assistant text and answer audio and is unchanged: each assistant
utterance is one sample, split by normalized answer text.

S2A follows the pinned MiniMind-O omni dataset contract
(``omni_dataset.py`` @ f900448c608318c53314ebf8a947ab05cd8c038e):

* one sample per conversation row; ``__getitem__`` picks an assistant turn at
  random on every read and then walks backwards turn by turn until the rendered
  prompt leaves the ``CONTEXT_MARGIN`` headroom, so long chats shorten instead
  of failing;
* the rendered prompt is truncated to ``max_seq_len`` (default ``512``) and
  then padded to exactly ``max_seq_len`` with the pad token;
* a random system prompt is prepended with ``SYSTEM_PROMPT_RATIO`` probability
  when the conversation has no leading system turn;
* the template-owned empty thinking scaffold is dropped with probability
  ``1 - EMPTY_THINK_KEEP_RATIO`` (retained 20% of the time);
* the assistant answer-body onset is searched only inside the first
  ``ONSET_SEARCH_POSITIONS`` positions after the assistant header, so a text
  ``</think>`` marker in history can never move the audio onset.

The dataset renders the prompt itself and exposes ``prompt_ids``,
``text_start`` (after the assistant header, before thinking), ``answer_start``
(audio body onset) and ``prompt_len`` on each sample; the trainer only consumes
those fields so a single random system/thinking decision is shared by the text
and audio supervision built from it. S2A splits group conversations by the
normalized first user text so every prefix of a conversation stays in one split
(generic first questions may conservatively group unrelated chats).

TTS keeps its own split key (normalized answer text). Empty/whitespace first
user strings are retained and share the normalized empty split key; missing
users and non-string content remain invalid. Selected targets stay in memory as
uint16; no cropping, prompt injection, or codec-token remapping is performed. A
full unlimited train view can therefore require substantial RAM: S2A now retains
every assistant's codes of a kept conversation, not only the last one.
Validation is strict for all scanned conversation pairings/text and for the
audio of a kept row; ``limit`` stops scanning early.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import unicodedata

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.multitask_audio_dataset import MultiTaskAudioSample

# Pinned MiniMind-O prompt augmentation constants (omni_dataset.py).
SYSTEM_PROMPTS = (
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
)
SYSTEM_PROMPT_RATIO = 0.2
EMPTY_THINK = "<think>\n\n</think>\n\n"
EMPTY_THINK_KEEP_RATIO = 0.2
THINK_END = "</think>\n\n"
ONSET_SEARCH_POSITIONS = 50
CONTEXT_MARGIN = 100
DEFAULT_MAX_SEQ_LEN = 512


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text_split(text: str, seed: int) -> str:
    key = f"{seed}\0{_normalized_text(text)}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    return "train" if bucket < 98 else "dev" if bucket == 98 else "test"


def _text_language(text: str) -> str:
    """Presence heuristic, not language identification (Latin includes accents).

    CJK unified/compatibility ideographs count as Chinese; Unicode letters
    whose names contain LATIN count as English. Both yields mixed, neither
    yields other. Japanese Kanji are consequently also classified as zh.
    """
    cjk = latin = False
    for char in text:
        name = unicodedata.name(char, "")
        cjk |= name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))
        latin |= char.isalpha() and "LATIN" in name
    return "mixed" if cjk and latin else "zh" if cjk else "en" if latin else "other"


def _tokenize(tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def answer_body_start(tokenizer, ids: list[int], assistant_start: int,
                      assistant_end: int | None = None) -> int:
    """Search 50 candidate positions, bounded by the assistant's end (upstream)."""
    marker = _tokenize(tokenizer, THINK_END)
    if not marker:
        raise ValueError("tokenizer returned an empty thinking-end marker")
    end = len(ids) if assistant_end is None else min(assistant_end, len(ids))
    for position in range(assistant_start, min(assistant_start + ONSET_SEARCH_POSITIONS, end)):
        if ids[position:position + len(marker)] == marker:
            return position + len(marker)
    return assistant_start


def _augment_system(messages: list[dict]) -> list[dict]:
    """Upstream ``pre_processing_chat``: 20% chance of a random leading system turn."""
    if messages and messages[0].get("role") != "system" and random.random() < SYSTEM_PROMPT_RATIO:
        return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + [dict(m) for m in messages]
    return [dict(m) for m in messages]


def render_conversation_ids(tokenizer, messages: list[dict], *,
                            max_seq_len: int | None = None,
                            include_text_start: bool = False):
    """Render a conversation and locate the answer-body onset for its last turn.

    Upstream ``post_processing_chat`` removes the empty thinking scaffold with
    probability ``1 - EMPTY_THINK_KEEP_RATIO``; the same decision is applied to
    both the generation prefix and the full chat so the prefix check still
    validates the template boundary. A template whose generation header is not a
    strict prefix of the full chat fails instead of guessing a boundary.
    """
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(prefix_text, str) or not isinstance(full_text, str):
        raise ValueError("s2a chat template must support tokenize=False string rendering")
    if EMPTY_THINK in full_text and random.random() > EMPTY_THINK_KEEP_RATIO:
        prefix_text = prefix_text.replace(EMPTY_THINK, "")
        full_text = full_text.replace(EMPTY_THINK, "")
    prefix, full = _tokenize(tokenizer, prefix_text), _tokenize(tokenizer, full_text)
    if not prefix or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("s2a chat template generation prefix does not match full chat; cannot locate assistant boundary safely")
    # Text supervision starts immediately AFTER the assistant header, before
    # thinking. Only audio skips the thinking block. Scan the truncated window,
    # as upstream does after input_ids[:max_length].
    visible = full if max_seq_len is None else full[:max_seq_len]
    text_start = min(len(prefix), len(visible))
    eos_text = getattr(tokenizer, "eos_token", None)
    eos_ids = (_tokenize(tokenizer, eos_text + "\n") if eos_text
               else [tokenizer.eos_token_id])
    assistant_end = next((pos for pos in range(text_start, len(visible))
                          if visible[pos:pos + len(eos_ids)] == eos_ids), len(visible))
    onset = answer_body_start(tokenizer, visible, text_start, assistant_end)
    if include_text_start:
        return full, onset, text_start
    return full, onset


class MiniMindT2ADataset(Dataset):
    """98/1/1 deterministic train/dev/test partition grouped by normalized text.

    For ``tts`` each row is one selected assistant utterance and ``limit`` counts
    those utterances. For ``s2a`` each row is one kept conversation and ``limit``
    counts conversations; ``__getitem__`` picks the supervised assistant turn at
    random per read. ``rows`` contains metadata only (never numpy arrays), while
    ``_codes``/``_s2a_codes`` keep the compact flat frame-major targets.
    """

    def __init__(self, path: str | Path, task: str = "tts", split: str = "train",
                 lang_filter: str | None = None, limit: int = 0, seed: int = 7,
                 min_frames: int = 1, tokenizer=None, max_seq_len: int = DEFAULT_MAX_SEQ_LEN) -> None:
        if task not in {"tts", "s2a"}:
            raise ValueError(f"MiniMindT2ADataset supports only tts or s2a, got {task!r}")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"split must be train, dev, or test, got {split!r}")
        if lang_filter not in {None, "zh", "en"}:
            raise ValueError("lang_filter must be None, 'zh', or 'en' (exact presence-based match)")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames < 1:
            raise ValueError("min_frames must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "MiniMindT2ADataset requires pyarrow; install it with `python -m pip install pyarrow`."
            ) from exc

        self.split_policy = ("sha256_seed_NFKC_casefold_whitespace_first_user_98_1_1" if task == "s2a"
                             else "sha256_seed_NFKC_casefold_whitespace_text_98_1_1")
        self.path, self.task, self.split = Path(path), task, split
        self.lang_filter, self.seed, self.min_frames, self.epoch = lang_filter, seed, min_frames, 0
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len
        self.rows: list[dict] = []
        self._codes: list[np.ndarray] = []
        self._s2a_codes: list[list[np.ndarray]] = []
        self.stats = dict(rows_scanned=0, assistants_scanned=0, split_filtered=0,
                          language_filtered=0, selected=0)
        with pq.ParquetFile(self.path) as parquet:
            schema = parquet.schema_arrow
            for column in ("conversations", "answer_audios"):
                if schema.names.count(column) != 1:
                    raise ValueError(f"{self.path}: schema requires exactly one {column!r} column")
            conversation_type = schema.field("conversations").type
            if not (pa.types.is_string(conversation_type) or pa.types.is_large_string(conversation_type)):
                raise ValueError(f"{self.path}: conversations schema must be a JSON string, got {conversation_type}")
            audio_type = schema.field("answer_audios").type
            is_list = lambda t: pa.types.is_list(t) or pa.types.is_large_list(t)
            if not (is_list(audio_type) and is_list(audio_type.value_type)
                    and pa.types.is_integer(audio_type.value_type.value_type)):
                raise ValueError(f"{self.path}: answer_audios schema must be list<list<integer>>, got {audio_type}")
            for batch in parquet.iter_batches(batch_size=32, columns=["conversations", "answer_audios"]):
                conversations = batch.column("conversations")
                answers = batch.column("answer_audios")
                for batch_row in range(batch.num_rows):
                    source_row = self.stats["rows_scanned"]
                    self.stats["rows_scanned"] += 1
                    context = f"{self.path}: row {source_row}"
                    raw = conversations[batch_row].as_py()
                    try:
                        messages = json.loads(raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{context}: conversations must contain valid JSON messages") from exc
                    if not isinstance(messages, list) or any(
                        not isinstance(message, dict) or not isinstance(message.get("role"), str)
                        for message in messages
                    ):
                        raise ValueError(f"{context}: conversations must be a list of role/content objects")
                    assistant_texts = [message.get("content") for message in messages if message["role"] == "assistant"]
                    audio_scalar = answers[batch_row]
                    if not audio_scalar.is_valid:
                        raise ValueError(f"{context}: answer_audios is null; assistant/audio pairing required")
                    audios = audio_scalar.values
                    if len(assistant_texts) != len(audios):
                        raise ValueError(f"{context}: assistant/audio count mismatch: {len(assistant_texts)} assistants, {len(audios)} audios")
                    if not assistant_texts:
                        raise ValueError(f"{context}: no assistant/audio pairs")
                    # Check every assistant's text before selection or an early limit return.
                    for assistant_index, text in enumerate(assistant_texts):
                        if not isinstance(text, str) or not _normalized_text(text):
                            raise ValueError(f"{context}, assistant {assistant_index}: content must be nonempty text")
                    assistant_positions = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
                    first_user = next((m.get("content") for m in messages if m["role"] == "user"), None)
                    if task == "s2a":
                        if any(m["role"] not in {"system", "user", "assistant"}
                               or not isinstance(m.get("content"), str) for m in messages):
                            raise ValueError(f"{context}: s2a requires text-only system/user/assistant conversations")
                        if not isinstance(first_user, str):
                            raise ValueError(f"{context}: s2a requires a first user with string content for conversation-grouped splits")
                        if not any(m["role"] == "user" for m in messages[:assistant_positions[0]]):
                            raise ValueError(f"{context}: s2a assistant must follow user context")
                    for assistant_index, text in enumerate(assistant_texts):
                        self.stats["assistants_scanned"] += 1
                    if task == "s2a":
                        if _text_split(first_user, seed) != split:
                            self.stats["split_filtered"] += 1
                            continue
                        languages = [_text_language(text) for text in assistant_texts]
                        if lang_filter is not None and lang_filter not in languages:
                            self.stats["language_filtered"] += 1
                            continue
                        codes_per_assistant = []
                        for assistant_index, text in enumerate(assistant_texts):
                            codes_per_assistant.append(self._validated_codes(
                                f"{context}, assistant {assistant_index}", audios[assistant_index], min_frames))
                        self.rows.append(dict(source_row=source_row, first_user=first_user,
                                              messages=[dict(m) for m in messages],
                                              texts=list(assistant_texts), assistant_positions=assistant_positions,
                                              text=assistant_texts[-1], lang=languages[-1]))
                        self._s2a_codes.append(codes_per_assistant)
                        self.stats["selected"] += 1
                        if limit and len(self.rows) >= limit:
                            return
                        continue
                    for assistant_index, text in enumerate(assistant_texts):
                        if _text_split(text, seed) != split:
                            self.stats["split_filtered"] += 1
                            continue
                        lang = _text_language(text)
                        if lang_filter is not None and lang != lang_filter:
                            self.stats["language_filtered"] += 1
                            continue
                        where = f"{context}, assistant {assistant_index}"
                        self._codes.append(self._validated_codes(where, audios[assistant_index], min_frames))
                        self.rows.append(dict(text=text, lang=lang, source_row=source_row,
                                              assistant_index=assistant_index))
                        self.stats["selected"] += 1
                        if limit and len(self.rows) >= limit:
                            return

    @staticmethod
    def _validated_codes(where: str, audio, min_frames: int) -> np.ndarray:
        """Validate one frame-major flat target; never crops or pads."""
        if not audio.is_valid or len(audio.values) == 0:
            raise ValueError(f"{where}: audio codes must be nonempty")
        values = audio.values
        if values.null_count:
            raise ValueError(f"{where}: audio codes must be integers, not null")
        codes = values.to_numpy(zero_copy_only=False)
        if codes.ndim != 1 or codes.size % 8:
            raise ValueError(f"{where}: flat frame-major code length {codes.size} must be divisible by 8")
        if np.any(codes < 0) or np.any(codes >= 2048):
            raise ValueError(f"{where}: audio codes must be integers in [0, 2048); no token remapping is supported")
        frames = codes.size // 8
        if frames < min_frames:
            raise ValueError(f"{where}: audio has {frames} frames, below min_frames={min_frames}; targets are never cropped or padded")
        return codes.astype(np.uint16, copy=True)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch  # Selection keys and targets deliberately do not change.

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MultiTaskAudioSample:
        if self.task == "s2a":
            return self._s2a_sample(index)
        codes = torch.from_numpy(self._codes[index].reshape(-1, 8).T.copy()).long().contiguous()
        return MultiTaskAudioSample(task=self.task, codes=codes, text=self.rows[index]["text"], messages=None)

    def _s2a_sample(self, index: int) -> MultiTaskAudioSample:
        """Random assistant turn, backward length fallback, then fixed-length padding."""
        if self.tokenizer is None:
            raise ValueError("MiniMindT2ADataset task='s2a' requires a tokenizer to render conversation prompts")
        row = self.rows[index]
        messages, positions = row["messages"], row["assistant_positions"]
        codes_per_assistant = self._s2a_codes[index]
        selected = 0
        if len(positions) > 1:
            for candidate_index in range(random.randrange(len(positions)), -1, -1):
                selected = candidate_index
                trial = _augment_system(messages[:positions[candidate_index] + 1])
                trial_ids, _ = render_conversation_ids(self.tokenizer, trial)
                if len(trial_ids) + CONTEXT_MARGIN < self.max_seq_len:
                    break
        # Upstream renders again after selecting the prefix. Trial augmentation
        # is not reused; only the final render is shared by text/audio labels.
        augmented = _augment_system(messages[:positions[selected] + 1])
        ids, onset, text_start = render_conversation_ids(
            self.tokenizer, augmented, max_seq_len=self.max_seq_len, include_text_start=True)
        prompt_len = min(len(ids), self.max_seq_len)
        answer_start = min(onset, prompt_len)
        # Upstream keeps/truncates even an overflowing first turn; it may have
        # no remaining assistant targets rather than raising a dataset error.
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        codes = codes_per_assistant[selected]
        sample = MultiTaskAudioSample(
            task="s2a",
            codes=torch.from_numpy(codes.reshape(-1, 8).T.copy()).long().contiguous(),
            text=row["texts"][selected],
            messages=augmented,
        )
        sample.prompt_ids = list(ids[:prompt_len]) + [int(pad)] * (self.max_seq_len - prompt_len)
        sample.prompt_len = prompt_len
        sample.answer_start = answer_start
        sample.text_start = text_start
        return sample
