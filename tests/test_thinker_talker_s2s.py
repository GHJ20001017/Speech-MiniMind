"""CPU contracts for SenseVoice S2S; no pretrained downloads or audio training."""
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel

from dataset.thinker_talker_s2s import build_s2s_batch, speech_prompt, ThinkerTalkerS2SDataset
from model.qwen3_talker import Qwen3ThinkerTalker, AUDIO_PAD_ID
from model.speech_input import SpeechInputModel, validate_codes, save_s2s_checkpoint, load_s2s_checkpoint
from scripts.infer_thinker_talker_s2s import synthesize_s2s
from trainer.train_thinker_talker_s2s import batch_loss, preflight_dataset
from trainer.train_audio_multitask import load_multitask_checkpoint


class Tokenizer:
    pad_token_id = 0
    eos_token_id = im_end_id = 2
    all_special_ids = [0, 2, 3, 4, 5]

    def __len__(self):
        return 32

    def get_vocab(self):
        return {"<|audio_start|>": 3, "<|audio_pad|>": 4, "<|audio_end|>": 5}

    def __call__(self, text, add_special_tokens=False):
        import re
        markers = dict(self.get_vocab(), **{
            "<test_start>": 1, "<test_assistant>": 6,
            "<test_end>": 2, "<test_newline>": 7,
        })
        if any(marker in text for marker in markers) or "<think>\n\n</think>\n\n" in text:
            pieces = re.split("(" + "|".join(map(re.escape, [*markers, "<think>\n\n</think>\n\n"])) + ")", text)
            ids = []
            for piece in pieces:
                if piece in markers:
                    ids.append(markers[piece])
                elif piece == "<think>\n\n</think>\n\n":
                    ids.extend([8, 9])
                else:
                    ids.extend(10 + ord(c) % 10 for c in piece)
            return {"input_ids": ids}
        if text == "</think>\n\n":
            return {"input_ids": [8, 9]}
        return {"input_ids": [10 + ord(c) % 10 for c in text]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        import re
        content = "".join(m["content"] for m in (messages if add_generation_prompt else messages[:-1]))
        if not tokenize:
            text = "<test_start>" + content + "<test_assistant>"
            if not add_generation_prompt:
                text += "<think>\n\n</think>\n\n" + messages[-1]["content"] + "<test_end><test_newline>"
            return text
        pieces = re.split(r"(<\|audio_(?:start|pad|end)\|>)", content)
        tokens = []
        for piece in pieces:
            if piece in self.get_vocab():
                tokens.append(self.get_vocab()[piece])
            else:
                tokens.extend(self(piece)["input_ids"])
        prefix = [1] + tokens + [6]
        if add_generation_prompt:
            return prefix
        return prefix + [8, 9] + self(messages[-1]["content"])["input_ids"] + [2, 7]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))

    def save_pretrained(self, path):
        vocab = {str(i): i for i in range(32)}
        for marker, i in self.get_vocab().items():
            del vocab[str(i)]
            vocab[marker] = i
        fast = PreTrainedTokenizerFast(tokenizer_object=Backend(WordLevel(vocab, unk_token="31")),
                                       pad_token="0", eos_token="2", unk_token="31")
        fast.save_pretrained(path)


class MockSenseVoice:
    output_dim = 6

    def __init__(self):
        self._engine = torch.nn.Linear(1, self.output_dim)

    def eval(self):
        self._engine.eval()
        return self

    def encode(self, waveforms, lengths, sample_rate, feature_augment=False):
        assert sample_rate == 16000 and not torch.is_grad_enabled()
        assert not self._engine.training
        return self._engine(waveforms.unsqueeze(-1)), lengths


def tiny():
    torch.manual_seed(3)
    config = Qwen3Config(num_hidden_layers=2, hidden_size=16, intermediate_size=32,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=32, max_position_embeddings=128, attention_dropout=0.0,
                        tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return SpeechInputModel(Qwen3ThinkerTalker(Qwen3ForCausalLM(config),
                                               num_talker_layers=1, adapter_rank=4), MockSenseVoice())


def sample(frames=3, answer="yes", code=1):
    return {"speech_features": (torch.arange(frames * 6).reshape(frames, 6).float() * code).sin(),
            "waveform": torch.full((frames,), float(code)), "prompt_text": "question",
            "answer_codes": torch.full((8, 2), 17, dtype=torch.long), "answer_text": answer}


def forward(model, batch):
    return model.forward_streams(**{k: v for k, v in batch.items() if not k.endswith("targets")})


def test_no_speech_exact_parity_and_question_affects_both_paths():
    model = tiny().eval()
    ids = torch.tensor([[1, 3, 4, 5]])
    audio = torch.full((1, 4, 8), AUDIO_PAD_ID)
    with torch.no_grad():
        base = model.base.forward_streams(ids, audio)
        wrapped = model.forward_streams(ids, audio)
        assert torch.equal(base.text_hidden, wrapped.text_hidden)
        assert torch.equal(base.audio_hidden, wrapped.audio_hidden)
        a = forward(model, build_s2s_batch(Tokenizer(), [sample(code=1)], 128))
        b = forward(model, build_s2s_batch(Tokenizer(), [sample(code=2)], 128))
    assert not torch.allclose(a.text_hidden[:, -1], b.text_hidden[:, -1])
    assert not torch.allclose(model.thinker.lm_head(a.text_hidden), model.thinker.lm_head(b.text_hidden))
    assert not torch.allclose(a.audio_hidden[:, -1], b.audio_hidden[:, -1])


@pytest.mark.parametrize("codes", [torch.ones(8, 0, dtype=torch.long), torch.ones(7, 2, dtype=torch.long),
                                   torch.ones(8, 2), torch.ones(8, 2, dtype=torch.bool),
                                   torch.full((8, 2), -1), torch.full((8, 2), 2048)])
def test_strict_codes(codes):
    with pytest.raises(ValueError):
        validate_codes(codes)


def test_alignment_padding_no_leakage_and_length():
    tokenizer = Tokenizer()
    batch = build_s2s_batch(tokenizer, [sample(), sample(5, "longer")], 128)
    _, prefix, trailer, slots = speech_prompt(tokenizer, 3)
    p = len(prefix)
    assert torch.equal(batch["input_ids"][0, :p], torch.tensor(prefix))
    assert torch.equal(batch["speech_mask"][0, :p], slots)
    assert batch["text_targets"][0, :p].eq(-100).all()
    assert batch["text_targets"][0, p:p+3].tolist() == tokenizer("yes")["input_ids"]
    assert batch["text_targets"][0, p+3:p+5].tolist() == trailer
    assert batch["audio_inputs"][0, :p+1].eq(AUDIO_PAD_ID).all()
    assert batch["audio_targets"][0, :p+1].eq(-1).all()
    for q in range(8):
        assert batch["audio_targets"][0, p+1+q:p+3+q, q].tolist() == [17, 17]
        assert batch["audio_targets"][0, p+3+q, q] == 2050
        assert batch["audio_inputs"][0, p+3+q, q] == 2050
    padding = ~batch["attention_mask"].bool()
    assert padding.any()
    assert batch["text_targets"][padding].eq(-100).all()
    assert batch["audio_targets"][padding].eq(-1).all()
    assert not batch["speech_mask"][padding].any()
    changed = build_s2s_batch(tokenizer, [sample(answer="different")], 128)
    assert torch.equal(changed["input_ids"][0, :p], batch["input_ids"][0, :p])
    truncated = build_s2s_batch(tokenizer, [sample()], p)
    assert truncated["text_targets"].eq(-100).all()
    assert truncated["audio_targets"].eq(-1).all()
    with pytest.raises(ValueError, match="answer_text"):
        build_s2s_batch(tokenizer, [sample(answer=" ")], 128)


def test_frozen_base_gradients_reach_frontend_from_joint_loss():
    model = tiny().train()
    loss = batch_loss(model, build_s2s_batch(Tokenizer(), [sample()], 128), "cpu", 16)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.frontend.parameters())
    assert not model.base.training


def test_mask_validation():
    model = tiny()
    batch = build_s2s_batch(Tokenizer(), [sample()], 128)
    batch["speech_mask"][0, 0] = True
    with pytest.raises(ValueError, match="slot count"):
        forward(model, batch)
    batch["speech_mask"][0, 0] = False
    batch["attention_mask"][batch["speech_mask"]] = 0
    with pytest.raises(ValueError, match="attended"):
        forward(model, batch)


@pytest.mark.parametrize("tuning", ["audio_proj", "all"])
def test_real_checkpoint_roundtrip_and_missing_frontend_rejected(tmp_path, tuning):
    model = tiny().eval()
    model.set_tuning(tuning)
    path = tmp_path / "checkpoint"
    batch = build_s2s_batch(Tokenizer(), [sample()], 128)
    save_s2s_checkpoint(model, Tokenizer(), path, epoch=1)
    restored, _ = load_s2s_checkpoint(path, "cpu", encoder=MockSenseVoice())
    assert restored.tuning == tuning
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with torch.no_grad():
        # from_pretrained may select SDPA instead of the toy's eager backend.
        torch.testing.assert_close(forward(model, batch).text_hidden,
                                   forward(restored, batch).text_hidden, atol=1e-6, rtol=1e-5)
    base, _ = load_multitask_checkpoint(path, "cpu")
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in model.base.state_dict().items())
    with pytest.raises(ValueError, match="overwrite"):
        save_s2s_checkpoint(model, Tokenizer(), path)
    metadata = json.loads((path / "s2s_metadata.json").read_text())
    metadata["codec"] = "encodec"
    (path / "s2s_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible"):
        load_s2s_checkpoint(path, "cpu", encoder=MockSenseVoice())
    with pytest.raises(ValueError, match="requires"):
        load_s2s_checkpoint(tmp_path / "missing", "cpu")


def test_generation_conditions_question_only_in_prefix(monkeypatch):
    model = tiny().eval()
    calls = []
    original = model.forward_streams
    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(model, "forward_streams", record)
    monkeypatch.setattr("scripts.infer_s2a.sample_text", lambda logits: 2)
    monkeypatch.setattr("scripts.infer_s2a.sample_audio", lambda logits, history: 17 if len(history) < 10 else 2050)
    question = sample()["waveform"]
    codes, report = synthesize_s2s(model, Tokenizer(), question, 2, "cpu", 2)
    assert codes.shape == (8, 2)
    assert report["text_stop_reason"] == "im_end"
    assert calls[0][1]["speech_features"][0].shape == (3, 6)
    assert calls[0][0][1].eq(AUDIO_PAD_ID).all()
    assert all("speech_features" not in kwargs and "speech_mask" not in kwargs for _, kwargs in calls[1:])
    assert all(args[0].shape[1] == 1 for args, _ in calls[1:])


def test_s2s_stopped_streams_feed_stop_once_then_pad(monkeypatch):
    model = tiny().eval()
    feedback = []
    original = model.forward_streams

    def record(input_ids, audio_inputs, **kwargs):
        if input_ids.shape[1] == 1:
            feedback.append(audio_inputs[0, 0].tolist())
        return original(input_ids, audio_inputs, **kwargs)

    def audio(logits, history):
        active = sum(value != AUDIO_PAD_ID for value in history)
        return 17 if active < 2 else 2050

    monkeypatch.setattr(model, "forward_streams", record)
    monkeypatch.setattr("scripts.infer_s2a.sample_text", lambda logits: 2)
    monkeypatch.setattr("scripts.infer_s2a.sample_audio", audio)
    codes, report = synthesize_s2s(model, Tokenizer(), sample()["waveform"], 8, "cpu", 2)
    assert codes.shape == (8, 2)
    assert report["audio_stop_reason"] == "audio_stop"
    assert report["stop_steps"] == list(range(2, 10))
    for q in range(7):
        stream = [row[q] for row in feedback]
        stop = stream.index(2050)
        assert stream[:stop].count(17) == 2
        assert all(value == AUDIO_PAD_ID for value in stream[stop + 1:])


def test_preflight_checks_all_rows_with_context():
    class Pairs:
        manifest = "train.jsonl"

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return sample(answer="yes" if index == 0 else "")

    with pytest.raises(ValueError, match=r"train.jsonl: sample 2:.*answer_text"):
        preflight_dataset(Pairs(), Tokenizer(), 128, tiny())


def test_training_cli_updates_only_frontend_and_saves_epoch(tmp_path, monkeypatch):
    from trainer import train_thinker_talker_s2s as cli

    base = tiny().base
    before = {key: value.clone() for key, value in base.state_dict().items()}
    models = []
    saved = []

    import io
    import soundfile as sf
    import pyarrow as pa
    import pyarrow.parquet as pq
    wave = io.BytesIO()
    sf.write(wave, np.ones(3, dtype=np.float32) * 0.1, 16000, format="WAV")
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([dict(conversations=json.dumps([
        {"role": "user", "content": "question"}, {"role": "assistant", "content": "yes"}]),
        question_audios=[wave.getvalue()], answer_audios=[[17] * 16])]), parquet)
    monkeypatch.setattr("dataset.thinker_talker_s2s.augment_waveform", lambda w: w)
    monkeypatch.setattr("random.choices", lambda *args, **kwargs: ["audio"])

    def wrap(loaded, encoder, encoder_path):
        model = SpeechInputModel(loaded, encoder, encoder_path)
        models.append((model, {key: value.clone() for key, value in model.frontend.state_dict().items()}))
        return model

    monkeypatch.setattr(cli, "validate_init_checkpoint", lambda path: {"task": "s2a"})
    monkeypatch.setattr(cli, "load_multitask_checkpoint", lambda *args: (base, Tokenizer()))
    monkeypatch.setattr(cli, "SpeechInputModel", wrap)
    monkeypatch.setattr(cli, "load_encoder", lambda *args: MockSenseVoice())
    monkeypatch.setattr(cli, "save_s2s_checkpoint", lambda model, tokenizer, path, epoch: saved.append((path, epoch)))
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["train", "--data", str(parquet),
                                    "--init-checkpoint", str(tmp_path / "base"),
                                    "--output", str(output), "--device", "cpu",
                                    "--epochs", "1", "--max-seq-len", "128"])
    cli.main()
    model, frontend_before = models[0]
    assert all(torch.equal(value, model.base.state_dict()[key]) for key, value in before.items())
    assert any(not torch.equal(value, model.frontend.state_dict()[key])
               for key, value in frontend_before.items())
    assert saved == [(output / "model_epoch_001", 1)]
    assert (output / "run_config.json").is_file()
    with pytest.raises(SystemExit):
        cli.main()


def test_direct_parquet_multiturn_missing_audio_and_fallback(tmp_path, monkeypatch):
    import io
    import soundfile as sf
    import pyarrow as pa
    import pyarrow.parquet as pq
    wav = io.BytesIO()
    sf.write(wav, np.ones((80, 2), dtype=np.float32) * 0.1, 8000, format="WAV")
    turns = [{"role": "system", "content": "system"},
             {"role": "user", "content": "question one"},
             {"role": "assistant", "content": "answer one"},
             {"role": "user", "content": "question two"},
             {"role": "assistant", "content": "answer two"}]
    rows = [dict(conversations=json.dumps(turns), question_audios=[wav.getvalue()],
                 answer_audios=[[17] * 16], ref_audios=[9999], spk_emb=[9999.0])]
    path = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    ds = ThinkerTalkerS2SDataset(f"{path}, {path}", Tokenizer(), 512)
    assert len(ds) == 2
    monkeypatch.setattr("random.randint", lambda a, b: b)
    second = ds[0]
    assert second["history"] == turns[:-1] and second["answer_text"] == "answer two"
    assert second["waveform"] is None and second["answer_codes"] is None
    from dataset.thinker_talker_s2s import prepare_s2s_batch
    batch = prepare_s2s_batch(tiny(), Tokenizer(), [second], 512)
    assert batch["audio_targets"].eq(-1).all() and not batch["speech_mask"].any()
    assert (batch["text_targets"] != -100).sum() == len("answer two") + 2
    # Backwards fallback tests pure rendered text + 100, not expanded audio slots.
    short = ThinkerTalkerS2SDataset(path, Tokenizer(), 145)[0]
    assert short["answer_text"] == "answer one"
    assert short["waveform"].shape == (160,)
    assert short["answer_codes"].shape == (8, 2)
    assert short["answer_codes"].eq(17).all()
    monkeypatch.setattr("random.randint", lambda a, b: a)
    assert ds[0]["answer_text"] == "answer one"
    deterministic = ThinkerTalkerS2SDataset(path, Tokenizer(), 512, training=False)
    assert deterministic[0]["answer_text"] == "answer two"


def test_waveform_cli_uses_sensevoice_and_only_decodes_mimi(tmp_path, monkeypatch):
    import soundfile as sf
    from scripts import infer_thinker_talker_s2s as cli
    source = tmp_path / "question.wav"
    sf.write(source, np.zeros(160, dtype=np.float32), 16000)
    generated = torch.full((8, 2), 17, dtype=torch.long)
    calls = []

    class Codec:
        num_codebooks = 8
        codebook_size = 2048
        sample_rate = 24000

        def encode(self, *args):
            raise AssertionError("question must never be encoded with Mimi")

        def decode(self, codes, lengths):
            assert torch.equal(codes[0], generated)
            calls.append("decode")
            return torch.zeros((1, 240)), torch.tensor([240])

    def generate(model, tokenizer, waveform, *args):
        assert waveform.shape == (160,)
        calls.append("generate")
        return generated, {"generated_text": "answer"}

    monkeypatch.setattr(cli, "load_s2s_checkpoint", lambda *args, **kwargs: (object(), Tokenizer()))
    monkeypatch.setattr(cli, "build_frozen_audio_codec", lambda *args, **kwargs: Codec())
    monkeypatch.setattr(cli, "synthesize_s2s", generate)
    output = tmp_path / "result"
    monkeypatch.setattr("sys.argv", ["infer", "--checkpoint", str(tmp_path / "model"),
                                    "--audio", str(source), "--mimi-model", "local-mimi",
                                    "--output", str(output), "--device", "cpu"])
    cli.main()
    assert calls == ["generate", "decode"]
    assert sf.info(output / "answer.wav").samplerate == 24000
    assert json.loads((output / "answer.json").read_text())["generated_text"] == "answer"
    with pytest.raises(SystemExit):
        cli.main()  # never overwrite prior output


@pytest.mark.parametrize("tuning", ["audio_proj", "all"])
def test_modes_and_variable_waveform_lengths(tuning):
    from dataset.thinker_talker_s2s import prepare_s2s_batch
    model = tiny()
    model.set_tuning(tuning)
    model.train()
    batch = prepare_s2s_batch(model, Tokenizer(), [sample(3), sample(7)], 128)
    assert [len(x) for x in batch["speech_features"]] == [3, 7]
    assert batch["speech_mask"].sum(1).tolist() == [3, 7]
    loss = batch_loss(model, batch, "cpu", 16)
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in model.encoder._engine.parameters())
    assert not model.encoder._engine.training
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.frontend.parameters())
    assert model.base.training == (tuning == "all")
    assert all(p.requires_grad == (tuning == "all") for p in model.base.parameters())
    if tuning == "all":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.parameters())


@pytest.mark.parametrize("modality", ["audio", "text", "audio_text", "text_audio"])
def test_input_modalities_and_projector_text_only(modality):
    from dataset.thinker_talker_s2s import prepare_s2s_batch
    model = tiny()
    batch = prepare_s2s_batch(model, Tokenizer(), [sample()], 128, modality=modality)
    assert batch["speech_mask"].sum() == (0 if modality == "text" else 3)
    history, _, _, _ = speech_prompt(Tokenizer(), 3, "question", modality)
    content = history[0]["content"]
    if modality == "audio_text":
        assert content.endswith("question")
    elif modality == "text_audio":
        assert content.startswith("question")
    elif modality == "text":
        assert content == "question"
    else:
        assert "question" not in content
    loss = batch_loss(model, batch, "cpu", 16)
    assert torch.isfinite(loss)
    assert loss.requires_grad == (modality != "text")


def test_old_codec_frontend_rejected_before_base_load(tmp_path, monkeypatch):
    from model import speech_input
    (tmp_path / "speech_frontend.pt").write_bytes(b"old")
    (tmp_path / "s2s_metadata.json").write_text(json.dumps({"s2s_format_version": 1}))
    monkeypatch.setattr(speech_input, "load_multitask_checkpoint", lambda *args: pytest.fail("loaded old checkpoint"))
    with pytest.raises(ValueError, match="incompatible"):
        load_s2s_checkpoint(tmp_path, "cpu")


def test_sensevoice_uses_automodel_frontend_and_stays_frozen(monkeypatch):
    import sys
    from types import SimpleNamespace
    from model.frozen_encoder import SenseVoiceFrozenEncoder
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        def output_size(self):
            return 1
        def forward(self, features, lengths):
            assert not self.training and not torch.is_grad_enabled()
            return features * self.weight, lengths
    class Frontend(torch.nn.Module):
        dither = 1.0
        def forward(self, waveforms, lengths):
            return waveforms.unsqueeze(-1), lengths
    encoder, frontend = Encoder(), Frontend()
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kw:
        SimpleNamespace(model=SimpleNamespace(encoder=encoder), kwargs={"frontend": frontend})))
    adapter = SenseVoiceFrozenEncoder("local", "cpu")
    assert adapter._frontend is frontend
    adapter.train()
    states, lengths = adapter.encode(torch.ones(2, 5), torch.tensor([3, 5]))
    assert states.dtype == torch.float32 and not states.requires_grad
    assert lengths.tolist() == [3, 5]
    assert not encoder.training and not encoder.weight.requires_grad
    seen = []
    def augment(valid):
        seen.append(tuple(valid.shape))
        valid.zero_()
    states, lengths = adapter.encode(torch.ones(2, 5), torch.tensor([3, 5]), feature_augment=augment)
    assert seen == [(3, 1), (5, 1)]
    assert states[0, :3].eq(0).all() and states[0, 3:].eq(1).all()
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.encode(torch.ones(1, 5), torch.tensor([5]), 24000)


def test_prepare_parquet_preserves_waveforms_and_text_without_codec(tmp_path, monkeypatch):
    import io
    import soundfile as sf
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts import prepare_speech_to_speech as cli
    from dataset.thinker_talker_s2s import load_waveform
    records = []
    for rate in (8000, 24000):
        buffer = io.BytesIO()
        sf.write(buffer, np.ones((rate // 10, 2), dtype=np.float32) * 0.1, rate, format="WAV")
        records.append(dict(conversations=json.dumps([
            {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]),
            question_audios=[buffer.getvalue()], answer_audios=[[17] * 16 + [2050] * 8]))
    parquet = tmp_path / "pairs.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet)
    output = tmp_path / "pairs"
    monkeypatch.setattr("sys.argv", ["prepare", "--parquet", str(parquet), "--output", str(output),
                                    "--lang", "all", "--dev-rows", "1"])
    cli.main()
    for split in ("train", "dev"):
        row = json.loads((output / f"{split}.jsonl").read_text())
        assert "prompt_codes" not in row and row["prompt_text"] == "question"
        assert load_waveform(output / row["prompt_audio"]).shape == (1600,)
        assert np.load(output / row["answer_codes"]).shape == (8, 2)
        assert row["answer_text"] == "answer"  # legacy preparation remains optional, not trainer input
    with pytest.raises(SystemExit, match="already exists"):
        cli.main()


@pytest.mark.parametrize("component", ["text", "audio"])
def test_each_loss_reaches_input_projector(component):
    model = tiny().train()
    batch = build_s2s_batch(Tokenizer(), [sample()], 128)
    if component == "text":
        batch["audio_targets"].fill_(-1)
    else:
        batch["text_targets"].fill_(-100)
    batch_loss(model, batch, "cpu", 16).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.frontend.parameters())
    assert all(p.grad is None for p in model.base.parameters())


def test_padding_does_not_change_valid_prefix_outputs():
    model = tiny().eval()
    alone = build_s2s_batch(Tokenizer(), [sample(3)], 128)
    padded = build_s2s_batch(Tokenizer(), [sample(3), sample(8)], 128)
    with torch.no_grad():
        expected, actual = forward(model, alone), forward(model, padded)
    n = alone["input_ids"].shape[1]
    torch.testing.assert_close(expected.text_hidden[0], actual.text_hidden[0, :n], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(expected.audio_hidden[0], actual.audio_hidden[0, :n], atol=1e-6, rtol=1e-5)


def test_truncation_crops_features_and_never_retargets_old_assistant():
    s = sample(100)
    s["history"] = [{"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "question"}]
    batch = build_s2s_batch(Tokenizer(), [s], 40)
    slots = int(batch["speech_mask"].sum())
    assert 0 < slots < 100 and batch["speech_features"][0].shape == (slots, 6)
    assert batch["text_targets"].eq(-100).all()
    assert batch["audio_targets"].eq(-1).all()
    # A truncated answer retains only codes in range; never moves STOP earlier.
    _, prefix, _, _ = speech_prompt(Tokenizer(), 3)
    batch = build_s2s_batch(Tokenizer(), [sample()], len(prefix) + 3)
    assert batch["audio_targets"][0, -2:, 0].tolist() == [17, 17]
    assert not batch["audio_targets"].eq(2050).any()


def test_sampling_shifted_shared_audio_mask_and_qwen_guards(monkeypatch):
    from dataset.thinker_talker_s2s import scheduled_sampling
    batch = build_s2s_batch(Tokenizer(), [sample()], 128)
    monkeypatch.setattr(torch, "randint", lambda high, shape, **kw: torch.full(shape, high - 1, **kw))
    result = scheduled_sampling(batch, Tokenizer(), 1.0)
    eligible = torch.zeros(batch["input_ids"].shape, dtype=torch.bool)
    eligible[:, :-1] = (batch["audio_targets"][:, 1:] != -1).any(-1)
    assert result["audio_inputs"][eligible].eq(2111).all()  # entire vocab, all 8 including inactive layers
    assert torch.equal(result["audio_inputs"][~eligible], batch["audio_inputs"][~eligible])
    text = torch.zeros_like(eligible)
    text[:, :-1] = batch["text_targets"][:, 1:] != -100
    text &= ~torch.isin(batch["input_ids"], torch.tensor(Tokenizer.all_special_ids))
    text &= ~batch["speech_mask"]
    assert result["input_ids"][text].eq(31).all()
    assert torch.equal(result["input_ids"][~text], batch["input_ids"][~text])
    assert torch.equal(result["audio_targets"], batch["audio_targets"])
    assert torch.equal(result["text_targets"], batch["text_targets"])


@pytest.mark.parametrize("transform", range(7))
def test_upstream_waveform_transform_probabilities_and_ranges(monkeypatch, transform):
    from dataset.thinker_talker_s2s import augment_waveform
    calls = []
    draws = iter([0.0 if i == transform else 1.0 for i in range(7)])
    monkeypatch.setattr("random.random", lambda: next(draws))
    def uniform(a, b):
        calls.append((a, b))
        return (a + b) / 2
    monkeypatch.setattr("random.uniform", uniform)
    monkeypatch.setattr("random.randint", lambda a, b: a)
    monkeypatch.setattr("random.choice", lambda values: values[0])
    np.random.seed(42)
    wave = np.linspace(-0.5, 0.5, 17000, dtype=np.float32)
    out = augment_waveform(wave)
    assert np.isfinite(out).all() and out.dtype == np.float32 and abs(out).max() <= 1
    expected = {0: [(0.7, 1.6)], 1: [(0.001, 0.01)], 2: [(0.8, 1.2)],
                3: [], 4: [], 5: [(0.05, 0.2)], 6: [(0.003, 0.015)]}
    assert calls == expected[transform]
    if transform == 0:
        assert len(out) == int(len(wave) / 1.15)
    elif transform == 3:
        assert not out[:4000].any() and np.array_equal(out[4000:], wave[4000:])
    elif transform != 2:  # midpoint gain = 1
        assert not np.array_equal(wave, out)
    assert wave[0] == -0.5  # augmentation never mutates source


def test_fbank_masks_valid_time_and_frequency(monkeypatch):
    from dataset.thinker_talker_s2s import augment_fbank
    monkeypatch.setattr("random.random", lambda: 0.0)
    values = iter([64, 3, 10, 2])
    monkeypatch.setattr("random.randint", lambda a, b: next(values))
    x = augment_fbank(torch.ones(20, 560))
    assert x[:, 3:67].eq(0).all() and x[2:12].eq(0).all()
    assert x[0, :3].eq(1).all() and x[12:, 67:].eq(1).all()


def test_modality_mixture_weights_and_missing_waveform(monkeypatch):
    from dataset.thinker_talker_s2s import prepare_s2s_batch
    calls = []
    def choose(values, weights):
        calls.append((values, weights))
        return ["audio_text"]
    monkeypatch.setattr("random.choices", choose)
    monkeypatch.setattr("dataset.thinker_talker_s2s.augment_waveform", lambda w: w)
    a, b = sample(), sample()
    b["waveform"] = None
    batch = prepare_s2s_batch(tiny(), Tokenizer(), [a, b], 128, mixture=True)
    assert calls == [(("audio", "text", "audio_text", "text_audio"), (4, 2, 2, 2))] * 2
    assert batch["speech_mask"].sum(1).tolist() == [3, 0]


def test_v2_frontend_metadata_remains_loadable(tmp_path):
    from model.speech_input import LEGACY_S2S_METADATA
    model = tiny()
    path = tmp_path / "legacy"
    save_s2s_checkpoint(model, Tokenizer(), path)
    metadata = dict(LEGACY_S2S_METADATA, tuning="audio_proj", encoder_path=model.encoder_path, acoustic_dim=6)
    (path / "s2s_metadata.json").write_text(json.dumps(metadata))
    restored, _ = load_s2s_checkpoint(path, "cpu", encoder=MockSenseVoice())
    assert all(torch.equal(v, restored.frontend.state_dict()[k]) for k, v in model.frontend.state_dict().items())


@pytest.mark.parametrize("draw,added", [(0.0, True), (0.2, False), (0.9, False)])
def test_source_system_prompts_and_bypass(monkeypatch, draw, added):
    from copy import deepcopy
    from dataset.thinker_talker_s2s import pre_processing_chat, SYSTEM_PROMPTS
    monkeypatch.setattr("random.random", lambda: draw)
    monkeypatch.setattr("random.choice", lambda values: values[-1])
    history = [{"role": "user", "content": "question"}]
    before = deepcopy(history)
    out = pre_processing_chat(history)
    assert len(SYSTEM_PROMPTS) == 10
    assert out == ([{"role": "system", "content": SYSTEM_PROMPTS[-1]}] if added else []) + history
    assert history == before and out[-1] is not history[-1]
    for messages in ([{"role": "system", "content": "keep"}] + history,
                     [dict(history[0], tools=[{"name": "lookup"}])]):
        monkeypatch.setattr("random.random", lambda: pytest.fail("bypass must not draw"))
        assert pre_processing_chat(messages) == messages


@pytest.mark.parametrize("draw,strip", [(0.2, False), (0.21, True), (0.99, True)])
def test_training_thinking_history_offsets_and_default(monkeypatch, draw, strip):
    from copy import deepcopy
    from dataset.thinker_talker_s2s import EMPTY_THINK
    tokenizer = Tokenizer()
    row = sample()
    row["history"] = [{"role": "system", "content": "keep"},
                      {"role": "user", "content": "old question"},
                      {"role": "assistant", "content": EMPTY_THINK + "old answer"},
                      {"role": "user", "content": "new question"}]
    before = deepcopy(row["history"])
    monkeypatch.setattr("random.random", lambda: draw)
    batch = build_s2s_batch(tokenizer, [row], 512, training=True)
    _, prefix, trailer, slots = speech_prompt(tokenizer, 3, history=before,
                                             strip_empty_thinking=strip)
    p = len(prefix)
    assert torch.equal(batch["input_ids"][0, :p], torch.tensor(prefix))
    assert batch["text_targets"][0, :p].eq(-100).all()
    assert batch["text_targets"][0, p:p + 3].tolist() == tokenizer("yes")["input_ids"]
    assert batch["text_targets"][0, p + 3:p + 5].tolist() == trailer
    assert batch["audio_targets"][0, :p + 1].eq(-1).all()
    assert batch["audio_targets"][0, p + 1, 0] == 17
    assert torch.equal(batch["speech_mask"][0, :p], slots)
    assert prefix.count(8) == (0 if strip else 2)
    assert row["history"] == before
    monkeypatch.setattr("random.random", lambda: pytest.fail("evaluation must not draw"))
    deterministic = build_s2s_batch(tokenizer, [row], 512)
    _, default_prefix, _, _ = speech_prompt(tokenizer, 3, history=before)
    assert deterministic["input_ids"][0, :len(default_prefix)].tolist() == default_prefix


@pytest.mark.parametrize("tools", [False, True])
def test_training_system_injection_and_final_tools_bypass(tmp_path, monkeypatch, tools):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset.thinker_talker_s2s import SYSTEM_PROMPTS
    history = [{"role": "user", "content": "question"},
               {"role": "assistant", "content": "yes", "tools": [{"name": "lookup"}] if tools else []}]
    path = tmp_path / "tools.parquet"
    pq.write_table(pa.Table.from_pylist([{"conversations": json.dumps(history)}]), path)
    tokenizer = Tokenizer()
    row = ThinkerTalkerS2SDataset(path, tokenizer)[0]
    assert row["tools_present"] == tools
    row["speech_features"] = torch.zeros(0, 6)
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.choice", lambda choices: choices[0])
    batch = build_s2s_batch(tokenizer, [row], 512, training=True)
    expected = history[:-1] if tools else [{"role": "system", "content": SYSTEM_PROMPTS[0]}] + history[:-1]
    _, prefix, _, _ = speech_prompt(tokenizer, 0, "question", "text", expected)
    assert batch["input_ids"][0, :len(prefix)].tolist() == prefix
    assert row["history"] == history[:-1]


@pytest.mark.parametrize("mixed", [False, True])
def test_training_no_grad_batches_and_zero_update_guard(tmp_path, monkeypatch, capsys, mixed):
    from trainer import train_thinker_talker_s2s as cli
    model = tiny()
    monkeypatch.setattr(cli, "validate_init_checkpoint", lambda path: {"task": "s2s"})
    monkeypatch.setattr(cli, "load_s2s_checkpoint", lambda *a, **k: (model, Tokenizer()))
    monkeypatch.setattr(cli, "ThinkerTalkerS2SDataset", lambda *a, **k: [sample()] * (2 if mixed else 1))
    batch = build_s2s_batch(Tokenizer(), [sample()], 128)
    monkeypatch.setattr(cli, "prepare_s2s_batch", lambda *a, **k: batch)
    monkeypatch.setattr(cli, "scheduled_sampling", lambda batch, tokenizer: batch)
    parameter = next(model.frontend.parameters())
    losses = iter([torch.tensor(2.0), parameter.sum() * 0 + 4.0] if mixed else [torch.tensor(2.0)])
    def mock_loss(*args, return_components=False):
        loss = next(losses)
        if return_components:
            return loss, {"text_loss": loss.detach(), "audio_loss": loss.detach().new_zeros(())}
        return loss
    monkeypatch.setattr(cli, "batch_loss", mock_loss)
    saved, steps = [], []
    monkeypatch.setattr(cli, "save_s2s_checkpoint", lambda *a: saved.append(a))
    original_step = torch.optim.AdamW.step
    def step(self, *a, **k):
        steps.append(True)
        return original_step(self, *a, **k)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--init-checkpoint", str(tmp_path / "base"),
                                    "--output", str(tmp_path / "run"), "--device", "cpu", "--epochs", "1",
                                    "--max-seq-len", "128"])
    if mixed:
        cli.main()
        report = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert report["train_updates"] == 1 and report["train_loss"] == 3.0
        assert len(steps) == len(saved) == 1
    else:
        with pytest.raises(ValueError, match="zero optimizer updates"):
            cli.main()
        assert steps == saved == []



@pytest.mark.parametrize("value", ["0", "-1"])
def test_accumulation_cli_rejects_nonpositive(monkeypatch, value):
    from trainer import train_thinker_talker_s2s as cli
    monkeypatch.setattr("sys.argv", ["train", "--data", "mock", "--init-checkpoint", "mock",
                                    "--output", "mock", "--gradient-accumulation-steps", value])
    monkeypatch.setattr(cli, "train", lambda args: pytest.fail("invalid CLI reached training"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_accumulation_cli_default(monkeypatch):
    from trainer import train_thinker_talker_s2s as cli
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("sys.argv", ["train", "--data", "mock", "--init-checkpoint", "mock", "--output", "mock"])
    values = []
    monkeypatch.setattr(cli, "train", lambda args: values.append(args.gradient_accumulation_steps))
    cli.main()
    assert values == [1]
