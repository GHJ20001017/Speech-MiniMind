"""Thinker-Talker S2A inference API contract tests."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts import infer_s2a


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    im_end_id = 9

    scaffold = [20, 21, 22, 23]

    def __call__(self, text, *, add_special_tokens):
        assert text == "</think>\n\n" and not add_special_tokens
        return {"input_ids": [22, 23]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        ids = [1]
        for index, message in enumerate(messages):
            if index == len(messages) - 1 and message["role"] == "assistant" and not add_generation_prompt:
                break
            ids += [3 if message["role"] == "user" else 4]
            ids += [10 + (ord(c) % 10) for c in message["content"]]
        if add_generation_prompt:
            ids += [5]
        elif messages[-1]["role"] == "assistant":
            # The assistant header is shared with add_generation_prompt output.
            ids += [5] + self.scaffold + [10 + (ord(c) % 10) for c in messages[-1]["content"]]
            ids += [9, 8]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class Head(torch.nn.Module):
    def __init__(self, stop=False):
        super().__init__()
        self.stop = stop
        self.calls = 0

    def forward(self, hidden):
        assert torch.all(hidden == 1), "audio heads must consume Talker hidden states"
        self.calls += 1
        result = torch.zeros(hidden.shape[0], 2112)
        result[:, infer_s2a.AUDIO_STOP_ID if self.stop else 1] = 100
        return result


class FakeModel:
    def __init__(self, text_token=7, stop_audio=True):
        self.config = SimpleNamespace(max_position_embeddings=256)
        self.thinker = SimpleNamespace(lm_head=lambda hidden: self._text_logits(hidden, text_token))
        self.audio_streams = SimpleNamespace(heads=[Head(stop_audio) for _ in range(infer_s2a.NUM_CODEBOOKS)])
        self.calls = []
        self._cache = object()

    @staticmethod
    def _text_logits(hidden, token):
        assert torch.all(hidden == 0), "text head must consume Thinker hidden states"
        logits = torch.zeros(hidden.shape[0], 32)
        logits[:, token] = 100
        return logits

    def get_input_embeddings(self):
        return lambda ids: torch.zeros(*ids.shape, 4)

    def forward_streams(self, input_ids, audio_inputs, attention_mask=None,
                        past_key_values=None, use_cache=False):
        self.calls.append((input_ids.detach().clone(), audio_inputs.detach().clone(),
                           past_key_values, use_cache))
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        return SimpleNamespace(text_hidden=hidden, audio_hidden=torch.ones_like(hidden),
                               past_key_values=self._cache)


def test_teacher_uses_forward_streams_and_first_answer_text():
    model = FakeModel()
    messages = [{"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}]
    infer_s2a.synthesize_s2a(model, Tokenizer(), messages, max_frames=1, device="cpu")
    assert model.calls[0][0][0, -1].item() == 10 + (ord("a") % 10)
    assert torch.all(model.calls[0][1] == 2049)
    assert model.calls[0][3] is True
    assert all(call[2] is model._cache for call in model.calls[1:])


def test_joint_has_no_reference_and_appends_trailer_at_text_cap():
    model = FakeModel(text_token=7)
    history = [{"role": "user", "content": "question"}]
    codes, metrics = infer_s2a.synthesize_s2a_joint(
        model, Tokenizer(), history, max_frames=1, device="cpu", max_text_tokens=1
    )
    with pytest.raises(ValueError, match="omit the final assistant"):
        infer_s2a.synthesize_s2a_joint(
            model, Tokenizer(), history + [{"role": "assistant", "content": "reference"}],
            max_frames=1, device="cpu", max_text_tokens=1,
        )
    # After the one generated token, the input sequence receives im_end/newline.
    seen = torch.cat([call[0].reshape(-1) for call in model.calls]).tolist()
    assert 9 in seen and 8 in seen
    assert metrics["text_stop_reason"] == "max_text_tokens"
    assert codes.shape[0] == infer_s2a.NUM_CODEBOOKS


def test_joint_audio_runs_after_text_end_and_pad_before_audio():
    model = FakeModel(text_token=9, stop_audio=False)
    codes, metrics = infer_s2a.synthesize_s2a_joint(
        model, Tokenizer(), [{"role": "user", "content": "q"}],
        max_frames=3, device="cpu", max_text_tokens=8,
    )
    assert codes.shape == (8, 3)
    assert metrics["text_stop_reason"] == "im_end"
    assert metrics["audio_stop_reason"] == "max_new_frames"
    assert model.calls[1][0].item() == 9
    assert model.calls[1][1].eq(2049).all()  # first text token has no audio yet
    assert model.calls[2][1][0, 0, 0] == 1  # q0 begins after first text token
    assert all(((call[1] < 2048) | (call[1] == 2049)).all() for call in model.calls)


def test_joint_text_runs_after_all_audio_stops():
    model = FakeModel(text_token=7, stop_audio=True)
    _, metrics = infer_s2a.synthesize_s2a_joint(
        model, Tokenizer(), [{"role": "user", "content": "q"}],
        max_frames=1, device="cpu", max_text_tokens=15,
    )
    assert metrics["generated_text_tokens"] == 15
    assert metrics["audio_stop_reason"] == "audio_stop"
    assert all(head.calls > 1 for head in model.audio_streams.heads)
    assert any(call[1].eq(2050).any() for call in model.calls)


@pytest.mark.parametrize("mode", ["teacher", "joint"])
@pytest.mark.parametrize("special", [2048, 2049, 2050, 2051, 2111])
def test_first_special_ends_output_but_later_samples_are_fed_back(mode, special):
    class ScheduledHead(Head):
        def forward(self, hidden):
            token = special if self.calls == 1 else 10 + self.calls
            self.calls += 1
            logits = hidden.new_full((hidden.shape[0], 2112), -100.0)
            logits[:, token] = 100
            return logits

    model = FakeModel(text_token=9)
    model.audio_streams.heads = [ScheduledHead() for _ in range(8)]
    history = [{"role": "user", "content": "q"}]
    if mode == "teacher":
        codes, metrics = infer_s2a.synthesize_s2a(
            model, Tokenizer(), history + [{"role": "assistant", "content": "a"}], 4, "cpu")
        offset = 0
    else:
        codes, metrics = infer_s2a.synthesize_s2a_joint(
            model, Tokenizer(), history, 4, "cpu", max_text_tokens=3)
        offset = 1
    assert codes.tolist() == [[10]] * 8
    assert metrics["stop_steps"] == list(range(1, 9))
    assert model.calls[2 + offset][1][0, 0, 0] == special
    assert model.calls[3 + offset][1][0, 0, 0] == 12
    assert model.audio_streams.heads[0].calls > 2


def test_teacher_rejects_full_text_and_audio_budget_overflow():
    model = FakeModel()
    model.config.max_position_embeddings = 24
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a" * 25}]
    with pytest.raises(ValueError, match="full teacher conversation"):
        infer_s2a.synthesize_s2a(model, Tokenizer(), messages, 1, "cpu")
    messages[-1]["content"] = "a"
    with pytest.raises(ValueError, match="context capacity"):
        infer_s2a.synthesize_s2a(model, Tokenizer(), messages, 25, "cpu")
    assert not model.calls


def test_joint_prefills_only_template_owned_scaffold():
    class RecordingTokenizer(Tokenizer):
        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, **kwargs):
            self.messages.append([dict(message) for message in messages])
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = RecordingTokenizer()
    model = FakeModel(text_token=9, stop_audio=False)
    history = [{"role": "user", "content": "q"}]
    _, metrics = infer_s2a.synthesize_s2a_joint(
        model, tokenizer, history, 1, "cpu", max_text_tokens=2)
    assert tokenizer.messages == [history, history + [{"role": "assistant", "content": ""}]]
    assert model.calls[0][0][0].tolist() == [1, 3, 13, 5] + tokenizer.scaffold
    assert model.calls[0][1].eq(infer_s2a.AUDIO_PAD_ID).all()
    assert metrics["answer_start"] == 8
    assert metrics["generated_text_tokens"] == 0  # scaffold is not sampled text
    assert history == [{"role": "user", "content": "q"}]


@pytest.mark.parametrize("scaffold", [[], Tokenizer.scaffold])
def test_teacher_onset_with_and_without_empty_thinking(scaffold):
    tokenizer = Tokenizer()
    tokenizer.scaffold = scaffold
    model = FakeModel(stop_audio=False)
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ab"}]
    _, metrics = infer_s2a.synthesize_s2a(model, tokenizer, messages, 1, "cpu")
    assert metrics["answer_start"] == 4 + len(scaffold)
    assert model.calls[0][0][0].tolist() == [1, 3, 13, 5] + scaffold + [17]
    assert model.calls[1][0].item() == 18
    assert model.calls[1][1][0, 0, 0].item() == 1


@pytest.mark.parametrize("relative, expected", [(0, 2), (49, 51), (50, 0)])
def test_thinking_onset_search_first_50_positions(relative, expected):
    # A marker in history cannot affect the selected assistant boundary.
    ids = [22, 23, 5] + [7] * relative + [22, 23, 17]
    assert infer_s2a._answer_body_start(Tokenizer(), ids, 3) == 3 + expected


def test_joint_scaffold_is_included_in_context_budget():
    model = FakeModel()
    model.config.max_position_embeddings = 16  # bare header fits, scaffold does not
    with pytest.raises(ValueError, match="exceed model context"):
        infer_s2a.synthesize_s2a_joint(
            model, Tokenizer(), [{"role": "user", "content": "q"}],
            1, "cpu", max_text_tokens=1)
    assert not model.calls


def test_joint_rejects_unaligned_empty_assistant_scaffold():
    tokenizer = Tokenizer()
    tokenizer.scaffold = [20, 21]
    model = FakeModel()
    with pytest.raises(ValueError, match="training answer-body onset"):
        infer_s2a.synthesize_s2a_joint(
            model, tokenizer, [{"role": "user", "content": "q"}],
            1, "cpu", max_text_tokens=1)
    assert not model.calls
