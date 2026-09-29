"""S2A conversation sampling, onset/truncation, simultaneous streams, and legacy regressions.

The S2A contract now follows MiniMind-O's omni dataset: one sample per
conversation, a random assistant turn per read, a backward length fallback when
``render + 100 >= max_seq_len``, fixed-length truncate/pad, 20% random system
augmentation, 20% empty-thinking retention, and the answer-body onset after
``</think>\\n\\n`` inside the first 50 positions. These tests pin that contract
instead of the retired per-assistant "full preservation or fail" behaviour.
"""
import json
import random
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dataset import minimind_t2a_dataset as t2a
from dataset.minimind_t2a_dataset import (CONTEXT_MARGIN, EMPTY_THINK, MiniMindT2ADataset,
                                          THINK_END, _text_split, answer_body_start,
                                          render_conversation_ids)
from dataset.multitask_audio_dataset import MultiTaskAudioSample
from model.qwen3_talker import Qwen3ThinkerTalker
from trainer import train_audio_multitask as training

MAX_SEQ_LEN = 512


class ChatTokenizer:
    """Qwen3-shaped fixture: the final assistant turn owns an empty thinking scaffold."""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + ord(char) % 100 for char in text]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not kwargs, kwargs
        assert not tokenize, "the dataset renders strings so it can drop the empty thinking block"
        out = ""
        for index, message in enumerate(messages):
            role, content = message["role"], message["content"]
            out += f"<|{role}|>"
            if role == "assistant":
                out += "\n" + (EMPTY_THINK if index == len(messages) - 1 else "") + content
            else:
                out += "\n" + content
            out += "<|end|>"
        if add_generation_prompt:
            out += "<|assistant|>\n"
        return out


TOKENIZER = ChatTokenizer()
SPEC = SimpleNamespace(audio_bos_id=6, audio_eos_id=7)


class BrokenTokenizer(ChatTokenizer):
    """Generation header that is not a prefix of the full chat."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs)
        return rendered + ("<|extra|>" if add_generation_prompt else "")


def in_split(split="train", seed=7, prefix="question"):
    return next(f"{prefix} {i}" for i in range(50000) if _text_split(f"{prefix} {i}", seed) == split)


def make_conversation(texts, first_user="q", system=None, frames=3):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": first_user})
    for index, text in enumerate(texts):
        messages.append({"role": "assistant", "content": text})
        if index != len(texts) - 1:
            messages.append({"role": "user", "content": f"follow up {index}"})
    audios = [[(index + 1)] * (8 * frames) for index in range(len(texts))]
    return messages, audios


def write_parquet(tmp_path, rows, name="s2a.parquet"):
    path = tmp_path / name
    schema = pa.schema([("conversations", pa.string()), ("answer_audios", pa.list_(pa.list_(pa.int64())))])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def dataset_for(tmp_path, texts, *, frames=3, max_seq_len=MAX_SEQ_LEN, split="train", seed=7,
                system=None, name="s2a.parquet"):
    messages, audios = make_conversation(texts, first_user=in_split(split, seed), system=system, frames=frames)
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": audios}], name)
    return MiniMindT2ADataset(path, "s2a", split, seed=seed, tokenizer=TOKENIZER, max_seq_len=max_seq_len)


def batch(sample, length=MAX_SEQ_LEN):
    return training.build_multitask_batch(TOKENIZER, SPEC, [sample], length)


def prompt_prefix_len(sample):
    """Length of the generation prefix; the empty-thinking decision never changes it."""
    text = TOKENIZER.apply_chat_template(sample.messages[:-1], tokenize=False, add_generation_prompt=True)
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def test_s2a_sample_unit_is_the_conversation_not_the_assistant(tmp_path):
    dataset = dataset_for(tmp_path, ["first reply", "second reply", "third reply"])
    assert len(dataset) == 1
    assert dataset.rows[0]["source_row"] == 0
    assert [row["text"] for row in dataset.rows] == ["third reply"]
    assert dataset.rows[0]["messages"][-1]["content"] == "third reply"
    assert dataset.stats["selected"] == 1 and dataset.stats["assistants_scanned"] == 3


def test_random_assistant_selection_covers_every_turn(tmp_path):
    texts = ["alpha reply", "beta reply", "gamma reply"]
    dataset = dataset_for(tmp_path, texts)
    random.seed(11)
    seen = {}
    for _ in range(400):
        sample = dataset[0]
        seen.setdefault(sample.text, set()).add(int(sample.codes[0, 0]))
    assert set(seen) == set(texts), seen
    for text, values in seen.items():
        assert values == {texts.index(text) + 1}


def test_every_read_is_an_independent_choice_and_messages_are_copied(tmp_path):
    dataset = dataset_for(tmp_path, ["alpha reply", "beta reply"])
    random.seed(3)
    outputs = {dataset[0].text for _ in range(200)}
    assert outputs == {"alpha reply", "beta reply"}
    sample = dataset[0]
    sample.messages[0]["content"] = "mutated"
    assert dataset[0].messages[0]["content"] != "mutated"
    assert dataset.rows[0]["messages"][0]["content"] != "mutated"


def test_backward_fallback_reduces_turns_until_the_render_fits(tmp_path):
    dataset = dataset_for(tmp_path, ["a" * 120, "b" * 120, "c" * 120])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(random, "randrange", lambda stop: stop - 1)
    try:
        sample = dataset[0]
    finally:
        monkeypatch.undo()
    chosen = dataset.rows[0]["texts"].index(sample.text)
    assert chosen < 2, "the longest turns cannot fit, so the fallback must shorten the conversation"
    ids, onset = render_conversation_ids(TOKENIZER, sample.messages)
    assert len(ids) + CONTEXT_MARGIN < MAX_SEQ_LEN
    assert sample.prompt_len == len(ids) < MAX_SEQ_LEN
    assert torch.all(sample.codes == chosen + 1)


def test_upstream_length_trial_is_separate_from_final_render(tmp_path, monkeypatch):
    calls = []
    original = t2a.render_conversation_ids

    def record(*args, **kwargs):
        calls.append(kwargs.get("include_text_start", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(t2a, "render_conversation_ids", record)
    monkeypatch.setattr(random, "random", lambda: 0.9)
    monkeypatch.setattr(random, "randrange", lambda stop: 0)
    dataset_for(tmp_path, ["a", "b"])[0]
    assert calls == [False, True]
    calls.clear()
    dataset_for(tmp_path, ["a"], name="one.parquet")[0]
    assert calls == [True], "upstream does not length-probe single-turn rows"


def test_fixed_length_truncation_and_padding(tmp_path):
    truncated = dataset_for(tmp_path, ["x" * 4000])[0]
    assert truncated.prompt_len == MAX_SEQ_LEN
    assert len(truncated.prompt_ids) == MAX_SEQ_LEN
    assert truncated.answer_start < truncated.prompt_len
    short = dataset_for(tmp_path, ["short"], name="short.parquet")[0]
    assert len(short.prompt_ids) == MAX_SEQ_LEN
    assert 0 < short.prompt_len < MAX_SEQ_LEN
    assert short.prompt_ids[short.prompt_len:] == [TOKENIZER.pad_token_id] * (MAX_SEQ_LEN - short.prompt_len)
    header = TOKENIZER(TOKENIZER.apply_chat_template(
        short.messages[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    assert short.prompt_ids[:len(header)] == header, "kept tokens keep the exact generation prefix"


def test_system_augmentation_is_optional_and_drawn_from_the_pinned_pool(tmp_path):
    dataset = dataset_for(tmp_path, ["reply one", "reply two"])
    random.seed(5)
    draws = [dataset[0] for _ in range(400)]
    assert {sample.messages[0]["role"] for sample in draws} == {"system", "user"}
    seen_prompts = {sample.messages[0]["content"] for sample in draws if sample.messages[0]["role"] == "system"}
    assert seen_prompts and seen_prompts <= set(t2a.SYSTEM_PROMPTS)


def test_empty_thinking_is_retained_only_fraction_of_the_time(tmp_path):
    dataset = dataset_for(tmp_path, ["reply"])
    random.seed(9)
    samples = [dataset[0] for _ in range(400)]
    kept = sum(sample.answer_start > sample.text_start for sample in samples)
    assert 0 < kept < 400, kept


def test_deterministic_system_and_thinking_decisions(monkeypatch):
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]
    monkeypatch.setattr(random, "random", lambda: 0.0)
    augmented = t2a._augment_system(messages)
    assert augmented[0]["role"] == "system" and augmented[0]["content"] in t2a.SYSTEM_PROMPTS
    assert messages[0]["role"] == "user", "inputs must not be mutated in place"
    ids_kept, onset_kept = render_conversation_ids(TOKENIZER, messages)
    monkeypatch.setattr(random, "random", lambda: 0.9)
    assert t2a._augment_system(messages)[0]["role"] == "user"
    ids_dropped, onset_dropped = render_conversation_ids(TOKENIZER, messages)
    assert len(ids_kept) == len(ids_dropped) + len(EMPTY_THINK)
    assert onsets_separated(onset_kept, onset_dropped)


def onsets_separated(onset_kept, onset_dropped):
    """Keeping the scaffold moves the onset past ``</think>\\n\\n``; dropping it does not."""
    assert onset_kept > onset_dropped
    return True


@pytest.mark.parametrize("relative,delta", [(0, 10), (10, 20), (49, 59), (50, 0)])
def test_onset_search_window_boundaries(relative, delta):
    marker = TOKENIZER(THINK_END, add_special_tokens=False)["input_ids"]
    assert len(marker) == 10
    header = "<|user|>hi<|end|><|assistant|>\n"
    ids = TOKENIZER(header, add_special_tokens=False)["input_ids"]
    before = len(ids)
    ids += [7] * relative + marker + TOKENIZER("body", add_special_tokens=False)["input_ids"]
    assert answer_body_start(TOKENIZER, ids, before) == before + delta


def test_audio_onset_tracks_the_thinking_end_and_keeps_stop(tmp_path):
    dataset = dataset_for(tmp_path, ["one", "two reply"], frames=3)
    random.seed(21)
    sample = next(sample for sample in (dataset[0] for _ in range(200)) if sample.text == "two reply")
    ids, labels, audio, feedback, attention = batch(sample)
    assert ids.size(1) == MAX_SEQ_LEN
    assert attention.bool().all()
    assert ids[0, :sample.prompt_len].tolist() == sample.prompt_ids[:sample.prompt_len]
    assert (labels[0, :sample.text_start] == -100).all()
    assert labels[0, sample.text_start:sample.prompt_len].tolist() == sample.prompt_ids[sample.text_start:sample.prompt_len]
    assert (labels[0, sample.prompt_len:] == -100).all()
    frames = sample.codes.size(1)
    for q in range(8):
        onset = sample.answer_start + q + 1
        assert (audio[0, :onset, q] == -1).all()
        assert torch.equal(audio[0, onset:onset + frames, q], sample.codes[q])
        assert audio[0, onset + frames, q] == training.AUDIO_STOP_ID
        assert feedback[0, onset + frames, q] == training.AUDIO_STOP_ID
        assert torch.equal(feedback[0, onset:onset + frames, q], sample.codes[q])
        assert audio[:, 1:][0, onset - 1, q] == sample.codes[q, 0]
    assert int((audio >= 0).sum()) == 8 * (frames + 1)
    assert int((audio == training.AUDIO_STOP_ID).sum()) == 8
    assert (feedback[0, :sample.answer_start + 1] == 2049).all()


def test_retained_thinking_has_text_labels_but_audio_starts_after_it(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    sample = dataset_for(tmp_path, ["reply"], system="system")[0]
    assert sample.text_start == prompt_prefix_len(sample)
    assert sample.answer_start == sample.text_start + len(EMPTY_THINK)
    _, labels, audio, _, _ = batch(sample)
    assert labels[0, sample.text_start:sample.answer_start].tolist() == TOKENIZER(EMPTY_THINK)["input_ids"]
    assert (labels[0, :sample.text_start] == -100).all()
    assert (audio[0, :sample.answer_start + 1] == -1).all()
    assert audio[0, sample.answer_start + 1, 0] == sample.codes[0, 0]


def test_onset_search_does_not_cross_assistant_end():
    marker = TOKENIZER(THINK_END)["input_ids"]
    ids = [7] * 5 + marker
    assert answer_body_start(TOKENIZER, ids, 0, assistant_end=5) == 0
    assert answer_body_start(TOKENIZER, ids, 0, assistant_end=6) == 5 + len(marker)


def test_first_turn_overflow_without_answer_tokens_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.9)
    sample = dataset_for(tmp_path, ["reply"], system="x" * 1000)[0]
    assert sample.prompt_len == sample.text_start == sample.answer_start == MAX_SEQ_LEN
    _, labels, audio, _, _ = batch(sample)
    assert (labels == -100).all()
    assert (audio == -1).all()


def test_thinking_marker_cut_by_window_does_not_move_audio_onset(tmp_path, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    full = dataset_for(tmp_path, ["reply"], system="system")[0]
    cut = dataset_for(tmp_path, ["reply"], system="system", name="cut.parquet",
                      max_seq_len=full.answer_start - 1)[0]
    assert cut.answer_start == cut.text_start == full.text_start
    _, labels, audio, _, _ = batch(cut)
    assert (labels[0, cut.text_start:cut.prompt_len] != -100).all()
    assert audio[0, cut.answer_start + 1, 0] == cut.codes[0, 0]


def test_long_audio_is_truncated_without_error_and_stop_is_dropped(tmp_path):
    sample = dataset_for(tmp_path, ["reply"], frames=600)[0]
    ids, labels, audio, feedback, attention = batch(sample)
    assert ids.size(1) == MAX_SEQ_LEN
    assert int((audio == training.AUDIO_STOP_ID).sum()) == 0, "STOP no longer fits the fixed window"
    onset = sample.answer_start + 1
    expected = sum(max(0, min(sample.codes.size(1), MAX_SEQ_LEN - (onset + q))) for q in range(8))
    assert int((audio >= 0).sum()) == expected > 0
    for q in range(8):
        start = onset + q
        room = min(sample.codes.size(1), MAX_SEQ_LEN - start)
        if room <= 0:
            continue
        assert torch.equal(audio[0, start:start + room, q], sample.codes[q, :room])
        assert torch.equal(feedback[0, start:start + room, q], sample.codes[q, :room])
    assert (audio[0, :sample.answer_start + 1] == -1).all()
    assert (feedback[0] != -1).all()


def test_batch_width_is_the_fixed_conversation_length(tmp_path):
    dataset = dataset_for(tmp_path, ["first", "second"], frames=2)
    random.seed(7)
    first, second = dataset[0], dataset[0]
    combined = training.build_multitask_batch(TOKENIZER, SPEC, [first, second], MAX_SEQ_LEN)
    assert combined[0].shape == (2, MAX_SEQ_LEN)
    assert int(combined[4].sum()) == 2 * MAX_SEQ_LEN
    for index, sample in enumerate((first, second)):
        alone = batch(sample)
        for actual, expected in zip(combined, alone):
            assert torch.equal(actual[index], expected[0])


def test_s2a_sample_without_dataset_render_is_rejected():
    bare = MultiTaskAudioSample("s2a", torch.zeros(8, 2, dtype=torch.long), "answer")
    with pytest.raises(ValueError, match="missing dataset-rendered prompt_ids"):
        training.build_multitask_batch(TOKENIZER, SPEC, [bare], MAX_SEQ_LEN)
    incomplete = MultiTaskAudioSample("s2a", torch.zeros(8, 2, dtype=torch.long), "answer", messages=[])
    incomplete.prompt_ids = [1, 2, 3]
    incomplete.prompt_len = 3
    incomplete.answer_start = 4
    with pytest.raises(ValueError, match="missing dataset-rendered prompt_ids"):
        training.build_multitask_batch(TOKENIZER, SPEC, [incomplete], MAX_SEQ_LEN)


def test_model_context_below_fixed_length_still_fails_loudly(tmp_path):
    sample = dataset_for(tmp_path, ["reply"])[0]
    with pytest.raises(ValueError, match="model context limit exceeded"):
        batch(sample, MAX_SEQ_LEN - 1)


def tiny_talker(vocab=128, hidden=16, layers=6, talker_layers=2,
                initialize_from_thinker=True):
    thinker = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048,
    )).eval()
    return Qwen3ThinkerTalker(thinker, num_talker_layers=talker_layers,
                              initialize_from_thinker=initialize_from_thinker)


def small_sample(tmp_path, texts=("answer",), frames=4):
    dataset = dataset_for(tmp_path, list(texts), frames=frames)
    random.seed(1)
    return dataset[0]


def test_thinker_text_and_talker_audio_streams_stay_independent(tmp_path):
    torch.manual_seed(0)
    model = tiny_talker().eval()
    ids, _, _, audio, attention = batch(small_sample(tmp_path))
    with torch.inference_mode():
        out = model.forward_streams(ids, audio, attention_mask=attention, use_cache=False)
        assert out.text_hidden.shape == out.audio_hidden.shape == (1, ids.size(1), 16)
        assert out.past_key_values is None
        other = audio.clone()
        other[audio >= 0] = (other[audio >= 0] + 1) % 2048
        changed = model.forward_streams(ids, other, attention_mask=attention, use_cache=False)
        assert torch.equal(out.text_hidden, changed.text_hidden)
        assert not torch.equal(out.audio_hidden, changed.audio_hidden)
        cached = model.forward_streams(ids, audio, attention_mask=attention, use_cache=True)
        assert isinstance(cached.past_key_values, tuple) and len(cached.past_key_values) == 2
        assert cached.past_key_values[0] is not cached.past_key_values[1]
        torch.testing.assert_close(cached.audio_hidden, out.audio_hidden)


def test_forward_joint_gradients_and_train_only_noise(tmp_path, monkeypatch):
    torch.manual_seed(3)
    backbone = tiny_talker()
    clean = batch(small_sample(tmp_path, frames=4))
    snapshots = [tensor.clone() for tensor in clean]
    captured, captured_text = [], []
    original = backbone.forward_streams

    def capture(input_ids, audio_inputs, attention_mask=None, **kwargs):
        captured.append(audio_inputs.clone())
        captured_text.append(input_ids.clone())
        return original(input_ids, audio_inputs, attention_mask=attention_mask, **kwargs)

    monkeypatch.setattr(backbone, "forward_streams", capture)
    model = training.MultitaskForward(backbone, 64, 1.0).train()
    loss, count = model(*clean)
    assert torch.isfinite(loss)
    assert count == int((clean[1][:, 1:] != -100).sum() + (clean[2][:, 1:] != -1).sum())
    loss.backward()
    talker = backbone.audio_streams
    for parameter in [backbone.get_input_embeddings().weight, backbone.thinker.lm_head.weight,
                      talker.decoder.layers[0].self_attn.q_proj.weight,
                      talker.decoder.norm.weight,
                      talker.head_base.weight, talker.head_adapters[3][2].weight,
                      talker.embedding_base.weight, talker.embedding_adapters[2][2].weight,
                      talker.semantic_projection[0].weight, talker.codec_projection[0].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    eligible = torch.zeros_like(clean[3], dtype=torch.bool)
    eligible[:, :-1] = (clean[2][:, 1:] != -1).any(-1, keepdim=True)
    assert torch.equal(captured[0][~eligible], clean[3][~eligible])
    assert ((captured[0][eligible] >= 0) & (captured[0][eligible] < 2112)).all()
    assert (captured[0][eligible] != clean[3][eligible]).any()
    text_eligible = torch.zeros_like(clean[0], dtype=torch.bool)
    text_eligible[:, :-1] = clean[1][:, 1:] != -100
    assert torch.equal(captured_text[0][~text_eligible], clean[0][~text_eligible])
    assert (captured_text[0][text_eligible] != clean[0][text_eligible]).any()
    model.eval()
    model(*clean)
    assert torch.equal(captured[-1], clean[3])
    assert torch.equal(captured_text[-1], clean[0])
    assert all(torch.equal(a, b) for a, b in zip(clean, snapshots))


def test_missing_conversations_and_incompatible_template_fail(tmp_path):
    with pytest.raises(ValueError, match="conversation Parquet"):
        training.load_stage_dataset(tmp_path, "s2a", "train")
    with pytest.raises(ValueError, match="generation prefix"):
        render_conversation_ids(BrokenTokenizer(), [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": "a"}])
    messages, audios = make_conversation(["reply"], first_user=in_split())
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": audios}], "bad.parquet")
    broken = MiniMindT2ADataset(path, "s2a", "train", tokenizer=BrokenTokenizer(), max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="generation prefix"):
        broken[0]


def test_missing_tokenizer_is_rejected_for_s2a(tmp_path):
    messages, audios = make_conversation(["reply"], first_user=in_split())
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": audios}], "nt.parquet")
    dataset = MiniMindT2ADataset(path, "s2a", "train")
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="requires a tokenizer"):
        dataset[0]


def test_padding_and_mixed_task_isolation(tmp_path):
    dataset = dataset_for(tmp_path, ["first", "longer reply"], frames=2)
    random.seed(13)
    a, b = dataset[0], dataset[0]
    legacy = MultiTaskAudioSample("tts", a.codes, "tts answer")
    combined = training.build_multitask_batch(TOKENIZER, SPEC, [a, b, legacy], MAX_SEQ_LEN)
    assert len(combined) == 5
    assert combined[0].shape == (3, MAX_SEQ_LEN)
    for index, sample in enumerate((a, b, legacy)):
        alone = batch(sample, MAX_SEQ_LEN)
        n = alone[0].size(1)
        for actual, expected in zip(combined, alone):
            assert torch.equal(actual[index, :n], expected[0])
        assert not combined[4][index, n:].any()
        assert (combined[1][index, n:] == -100).all()
        assert (combined[2][index, n:] == -1).all()
        assert (combined[3][index, n:] == 2049).all()


@pytest.mark.parametrize("messages", [
    [{"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": None}, {"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": 0}, {"role": "assistant", "content": "answer"}],
    [{"role": "user"}, {"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": ["multimodal"]}, {"role": "assistant", "content": "answer"}],
])
def test_invalid_parquet_conversation_fails_before_split_filter(tmp_path, messages):
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": [[0] * 8]}])
    with pytest.raises(ValueError, match="s2a requires"):
        MiniMindT2ADataset(path, "s2a")


@pytest.mark.parametrize("seed", [7, 19, 42])
def test_empty_user_prompts_retained_paired_and_grouped(tmp_path, seed):
    prompts = ["", " ", "\t\n  ", "\u3000\u00a0"]
    rows = []
    for index, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"Please provide your question {index}"}]
        rows.append({"conversations": json.dumps(messages), "answer_audios": [[index + 1] * 16]})
    path = write_parquet(tmp_path, rows, "empty.parquet")
    expected_split = _text_split("", seed)
    assert all(_text_split(prompt, seed) == expected_split for prompt in prompts)
    for split in ("train", "dev", "test"):
        dataset = MiniMindT2ADataset(path, "s2a", split, seed=seed, tokenizer=TOKENIZER, max_seq_len=MAX_SEQ_LEN)
        assert dataset.stats["rows_scanned"] == len(prompts)
        assert len(dataset) == (len(prompts) if split == expected_split else 0)
        if split != expected_split:
            continue
        assert [row["source_row"] for row in dataset.rows] == list(range(len(prompts)))
        for index, sample in enumerate(dataset):
            users = [message["content"] for message in sample.messages if message["role"] == "user"]
            assert users[0] == prompts[index]
            assert torch.all(sample.codes == index + 1)
            assert dataset.rows[index]["text"] == f"Please provide your question {index}"
            assert len(sample.prompt_ids) == MAX_SEQ_LEN


def test_language_filter_keeps_conversations_with_a_matching_turn(tmp_path):
    rows = []
    mixed, mixed_audios = make_conversation(["你好世界", "plain latin reply"], first_user=in_split())
    rows.append({"conversations": json.dumps(mixed), "answer_audios": mixed_audios})
    english, english_audios = make_conversation(["plain latin reply"], first_user=in_split())
    rows.append({"conversations": json.dumps(english), "answer_audios": english_audios})
    path = write_parquet(tmp_path, rows, "lang.parquet")
    both = MiniMindT2ADataset(path, "s2a", "train", tokenizer=TOKENIZER, max_seq_len=MAX_SEQ_LEN)
    assert len(both) == 2
    chinese = MiniMindT2ADataset(path, "s2a", "train", lang_filter="zh", tokenizer=TOKENIZER, max_seq_len=MAX_SEQ_LEN)
    assert len(chinese) == 1 and chinese.stats["language_filtered"] == 1
    random.seed(4)
    assert chinese[0].text in {"你好世界", "plain latin reply"}


def test_split_filter_groups_the_whole_conversation(tmp_path):
    messages, audios = make_conversation(["reply one", "reply two"], first_user=in_split("dev"))
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": audios}], "dev.parquet")
    train = MiniMindT2ADataset(path, "s2a", "train", tokenizer=TOKENIZER, max_seq_len=MAX_SEQ_LEN)
    dev = MiniMindT2ADataset(path, "s2a", "dev", tokenizer=TOKENIZER, max_seq_len=MAX_SEQ_LEN)
    assert len(train) == 0 and train.stats["split_filtered"] == 1
    assert len(dev) == 1


def test_legacy_batch_contract_and_s2a_stage_discovery(tmp_path):
    codes = small_sample(tmp_path).codes
    for task in ("tts", "audio_lm"):
        result = batch(MultiTaskAudioSample(task, codes, "answer", prompt_codes=codes), 4096)
        assert len(result) == 5
        assert (result[1] == -100).all()
    with pytest.raises(ValueError, match="ASR is unsupported"):
        batch(MultiTaskAudioSample("asr", codes, "answer"), 4096)
    assert training.parse_task("s2a") == "s2a"
    (tmp_path / "stage_04_s2a").mkdir()
    assert training.existing_stages(tmp_path) == [(4, "s2a")]
    assert training.next_stage_index(tmp_path) == 5


def test_load_stage_dataset_forwards_tokenizer_and_fixed_length(tmp_path):
    messages, audios = make_conversation(["reply"], first_user=in_split())
    path = write_parquet(tmp_path, [{"conversations": json.dumps(messages), "answer_audios": audios}], "stage.parquet")
    dataset = training.load_stage_dataset(path, "s2a", "train", tokenizer=TOKENIZER, max_seq_len=256)
    assert dataset.tokenizer is TOKENIZER and dataset.max_seq_len == 256
    assert "backward_fallback" in training.S2A_SEMANTICS["length_policy"]
    assert training.S2A_SEMANTICS["chat_template"].endswith("empty_thinking_removed_with_p0.8")
    sample = dataset[0]
    assert len(sample.prompt_ids) == 256
    assert np.asarray(sample.codes).shape[0] == 8
