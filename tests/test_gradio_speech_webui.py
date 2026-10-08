"""Model-free regressions for the Gradio speech UI and owned generation worker."""
import io
import queue
import sys
import threading
import types
import wave
from contextlib import nullcontext

import numpy as np
import pytest

from scripts import speech_webui_helpers as helpers
from scripts.visualize_asr_webui import (
    ASR_MODE, QA_MODE, WebUIRunner, build_demo, build_parser, greedy_collapse,
)


def wav_bytes(samples=None, rate=16000):
    samples = np.array([0, 100, -100] if samples is None else samples, dtype="<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


class FakeEngine:
    tts_enabled = True
    _tts_text = staticmethod(lambda text: text.strip())

    def __init__(self, fail=False, answers=None, stream_fail=False):
        self.spoken = []
        self.closed = False
        self.fail = fail
        self.stream_fail = stream_fail
        self.answers = (["第一句。", "第一句。第二句", "第一句。第二句"]
                        if answers is None else answers)

    def stream(self, *_):
        try:
            for answer in self.answers:
                assert not self.spoken  # Text generation must finish before TTS.
                yield answer
            assert not self.spoken
            if self.stream_fail:
                raise RuntimeError("text generation failure")
        finally:
            self.closed = True

    def synthesize(self, text):
        assert self.closed  # The source must also be closed before TTS starts.
        self.spoken.append(text)
        if self.fail:
            raise RuntimeError("synthetic failure")
        return wav_bytes([len(self.spoken)]), 16000


def test_incremental_text_then_single_complete_answer_audio():
    engine = FakeEngine()
    source = helpers.stream_answer(engine, np.ones(3), "", 0, 32)
    first = next(source)
    assert first[0] == "第一句。" and first[1] is None
    assert not engine.closed and not engine.spoken
    rows = [first, *source]
    assert engine.spoken == ["第一句。第二句"]
    assert all(row[1] is None for row in rows[:-1])
    assert any(row[0] == "第一句。第二句" for row in rows[:-1])
    assert [r[1] for r in rows if r[1] is not None] == [wav_bytes([1])]
    assert rows[-1] == ("第一句。第二句", wav_bytes([1]), "完成")
    assert engine.closed


def test_tts_error_keeps_text_and_is_not_retried():
    engine = FakeEngine(fail=True)
    rows = list(helpers.stream_answer(engine, np.ones(3), "", 0, 32))
    assert rows[-1][0] == "第一句。第二句"
    assert "synthetic failure" in rows[-1][2]
    assert all(r[1] is None for r in rows)
    assert engine.spoken == ["第一句。第二句"]
    assert engine.closed


def test_text_generation_error_keeps_text_without_synthesis(monkeypatch):
    patch_audio(monkeypatch)
    engine = FakeEngine(stream_fail=True)
    runner = WebUIRunner(engine=engine)
    rows = list(runner.run("input.wav", QA_MODE, "", 0, 32, "error"))
    assert rows[-1][2] == "第一句。第二句"
    assert "text generation failure" in rows[-1][4]
    assert all(r[3] is None for r in rows)
    assert not engine.spoken and engine.closed
    assert not runner._sessions and not runner._inference_lock.locked()


def test_close_and_cooperative_cancel_never_emit_stale_audio():
    engine = FakeEngine()
    source = helpers.stream_answer(engine, np.ones(3), "", 0, 32)
    next(source)
    source.close()
    assert engine.closed and not engine.spoken
    engine = FakeEngine()
    cancelled = threading.Event()
    original = engine.synthesize

    def synthesize(text):
        result = original(text)
        cancelled.set()  # Stop during non-interruptible TTS.
        return result

    engine.synthesize = synthesize
    rows = list(helpers.stream_answer(engine, np.ones(3), "", 0, 32, cancelled))
    assert all(r[1] is None for r in rows)
    assert rows[-1] == ("第一句。第二句", None, "已停止")
    assert engine.closed
    assert engine.spoken == ["第一句。第二句"]


def test_cancel_after_text_source_closes_skips_tts():
    cancelled = threading.Event()

    class CancelOnCloseEngine(FakeEngine):
        def stream(self, *args):
            try:
                yield from super().stream(*args)
            finally:
                cancelled.set()

    engine = CancelOnCloseEngine()
    rows = list(helpers.stream_answer(engine, np.ones(3), "", 0, 32, cancelled))
    assert rows[-1] == ("第一句。第二句", None, "已停止")
    assert all(r[1] is None for r in rows)
    assert engine.closed and not engine.spoken


def test_cancel_during_text_generation_skips_tts():
    engine = FakeEngine()
    cancelled = threading.Event()
    source = helpers.stream_answer(engine, np.ones(3), "", 0, 32, cancelled)
    first = next(source)
    cancelled.set()
    rows = [first, *source]
    assert rows[-1] == (first[0], None, "已停止")
    assert all(r[1] is None for r in rows)
    assert engine.closed and not engine.spoken


@pytest.mark.parametrize("answers", [[], [""], ["   \n\t"]])
def test_blank_answer_never_synthesizes(answers):
    engine = FakeEngine(answers=answers)
    rows = list(helpers.stream_answer(engine, np.ones(3), "", 0, 32))
    assert not rows[-1][0].strip()
    assert rows[-1][2] == "模型返回空回答"
    assert all(r[1] is None for r in rows)
    assert engine.closed and not engine.spoken


def test_splitter_and_ctc_boundaries():
    assert helpers.split_sentence_chunks("第一句。第二句") == (["第一句。"], "第二句")
    assert helpers.split_sentence_chunks("tail", final=True) == (["tail"], "")
    chunks, tail = helpers.split_sentence_chunks("x" * 205)
    assert list(map(len, chunks)) == [100, 100] and len(tail) == 5
    assert helpers.split_sentence_chunks("   ", final=True) == ([], "")
    with pytest.raises(ValueError):
        helpers.split_sentence_chunks("abc", max_chars=0)
    assert greedy_collapse([0, 1, 1, 0, 1, 2, 2], {1: "你", 2: "好"}) == "你你好"


def test_decoder_empty_invalid_resampling(monkeypatch):
    monkeypatch.setattr(helpers.shutil, "which", lambda _: None)
    assert helpers.decode_bytes_to_mono16k(wav_bytes(rate=8000)).shape == (6,)
    for value in [b"", wav_bytes([]), b"not wav"]:
        with pytest.raises(ValueError):
            helpers.decode_bytes_to_mono16k(value)


def test_synthesis_is_pcm16_and_preserves_amplitude():
    class TTS:
        def generate_custom_voice(self, **kwargs):
            assert kwargs == dict(text="你好", language="Chinese", speaker="Serena")
            return [np.array([0.1, -0.1])], 24000

    engine = helpers.SpeechEngine(None, None, None, None, "cpu", 512, TTS())
    data, rate = engine.synthesize("**你好**")
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getframerate() == rate == 24000
        assert wav.getsampwidth() == 2
        assert np.frombuffer(wav.readframes(2), dtype="<i2").tolist() == [3276, -3276]


def patch_audio(monkeypatch):
    monkeypatch.setattr(helpers.shutil, "which", lambda _: None)
    monkeypatch.setattr("pathlib.Path.read_bytes", lambda _: wav_bytes())


def test_asr_routing_and_request_local_state(monkeypatch):
    patch_audio(monkeypatch)
    offline = types.SimpleNamespace(transcribe=lambda *_: "真正转写")
    streaming = types.SimpleNamespace(iter_incremental=lambda *_: iter([(0.1, "转"), (0.2, "转写")]))
    runner = WebUIRunner(offline, streaming, feature_fn=lambda audio: audio)
    rows = list(runner.run("input.wav", ASR_MODE, "ignored", 0, 32, "one"))
    assert rows[-1][:4] == ("真正转写", "转写", "", None)
    assert [row[1] for row in rows[1:3]] == ["转", "转写"]
    assert not runner._sessions and not runner._inference_lock.locked()
    assert "请先录音" in list(runner.run(None, ASR_MODE, "", 0, 32, "two"))[-1][-1]
    assert "语音问答需要" in list(runner.run("in", QA_MODE, "", 0, 32, "two"))[-1][-1]


@pytest.mark.parametrize("fail", [False, True])
def test_qa_routing_and_exception_preserves_answer(monkeypatch, fail):
    patch_audio(monkeypatch)
    engine = FakeEngine(fail=fail)
    runner = WebUIRunner(engine=engine)
    rows = list(runner.run("input.wav", QA_MODE, "", 0, 32, "one"))
    assert rows[-1][2] == "第一句。第二句"
    assert engine.spoken == ["第一句。第二句"] and engine.closed
    assert all(r[3] is None for r in rows[:-1])
    if fail:
        assert rows[-1][3] is None
        assert "synthetic failure" in rows[-1][-1]
    else:
        assert rows[-1][3] == wav_bytes([1])
        assert rows[-1][-1] == "完成"
    assert all(not r[0] and not r[1] for r in rows)
    assert not runner._sessions and not runner._inference_lock.locked()


def test_concurrent_waiter_cancels_without_affecting_other_session(monkeypatch):
    patch_audio(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def transcribe(*_):
        entered.set()
        assert release.wait(3)
        return "owner"

    runner = WebUIRunner(offline=types.SimpleNamespace(transcribe=transcribe), feature_fn=lambda x: x)
    results = {}
    owner = threading.Thread(target=lambda: results.update(owner=list(runner.run("in", ASR_MODE, "", 0, 1, "owner"))))
    owner.start()
    assert entered.wait(2)
    waiting = runner.run("in", ASR_MODE, "", 0, 1, "waiting")
    next(waiting)  # registered, but not holding inference lock
    runner.cancel("waiting")
    assert list(waiting)[-1][-1] == "已停止"
    assert not runner._sessions["owner"].is_set()
    release.set()
    owner.join(3)
    assert not owner.is_alive() and results["owner"][-1][0] == "owner"
    assert not runner._sessions and not runner._inference_lock.locked()


def install_fake_transformers(monkeypatch):
    class Streamer:
        def __init__(self, *_args, **_kwargs):
            self.queue = queue.Queue()

        def __next__(self):
            item = self.queue.get(timeout=0.01)
            if item is None:
                raise StopIteration
            return item

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        StoppingCriteria=object, StoppingCriteriaList=list, TextIteratorStreamer=Streamer,
    ))


def test_generation_worker_close_stops_and_joins(monkeypatch):
    install_fake_transformers(monkeypatch)
    ended = threading.Event()

    class Model:
        config = types.SimpleNamespace(use_cache=True)

        def generate(self, streamer, stopping_criteria, **_kwargs):
            try:
                streamer.queue.put("hello")
                while not stopping_criteria[0](None, None):
                    ended.wait(0.001)
            finally:
                ended.set()

    tokenizer = types.SimpleNamespace(eos_token_id=1, pad_token_id=0)
    source = helpers.stream_generated_text(Model(), tokenizer, None, None, 5, 0, threading.Event())
    assert next(source) == "hello"
    source.close()
    assert ended.is_set()
    assert not any(t.name == "speech-webui-generate" for t in threading.enumerate())


def test_generation_worker_error_surfaces_without_timeout(monkeypatch):
    install_fake_transformers(monkeypatch)

    class Model:
        config = types.SimpleNamespace(use_cache=True)

        def generate(self, **_):
            raise RuntimeError("model failure")

    tokenizer = types.SimpleNamespace(eos_token_id=1, pad_token_id=None)
    with pytest.raises(RuntimeError, match="model failure"):
        list(helpers.stream_generated_text(Model(), tokenizer, None, None, 5, 0, threading.Event()))
    assert not any(t.name == "speech-webui-generate" for t in threading.enumerate())


def test_normal_generation_does_not_cancel_final_tts(monkeypatch):
    install_fake_transformers(monkeypatch)

    class Model:
        config = types.SimpleNamespace(use_cache=True)

        def generate(self, streamer, **_):
            streamer.queue.put("tail")
            streamer.queue.put(None)

    cancelled = threading.Event()
    tokenizer = types.SimpleNamespace(eos_token_id=1, pad_token_id=None)
    assert list(helpers.stream_generated_text(Model(), tokenizer, None, None, 5, 0, cancelled)) == ["tail"]
    assert not cancelled.is_set()


def test_gradio_wiring_and_cli_model_options(monkeypatch):
    components = []
    calls = []

    class Component:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            components.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def click(self, *args, **kwargs):
            calls.append((args, kwargs))

    fake = types.SimpleNamespace(**{name: Component for name in
                                   ("Blocks", "Markdown", "Radio", "Audio", "Textbox", "Row", "Slider", "Button")},
                                 Request=type("Request", (), {}), skip=lambda: {"__type__": "update"})
    monkeypatch.setitem(sys.modules, "gradio", fake)
    args = build_parser().parse_args([
        "--encoder-type", "sensevoice", "--sensevoice-model", "local/sv",
        "--projector-checkpoint", "p.pt", "--qwen3-model", "qwen",
        "--tts-model", "tts", "--tts-speaker", "Serena", "--ssl-auto",
    ])
    build_demo(args, WebUIRunner(engine=FakeEngine()))
    audio = next(c for c in components if c.kwargs.get("label") == "回答语音（完整音频）")
    assert audio.kwargs["streaming"] is False
    assert audio.kwargs["autoplay"] is True
    assert calls[0][1]["concurrency_limit"] is None
    assert calls[0][0][0].__annotations__["request"] is fake.Request
    assert calls[1][1]["queue"] is False
    callback = calls[0][0][0]
    rows = list(callback(None, QA_MODE, "", 0, 32, types.SimpleNamespace(session_hash="test")))
    assert rows[0][3] is None  # reset previous playback once
    assert rows[-1][3] == {"__type__": "update"}  # text-only updates do not reset audio
    assert calls[1][0][0](types.SimpleNamespace(session_hash="test"))[0] is None


def test_tls_argument_validation():
    args = build_parser().parse_args(["--ssl-keyfile", "key"])
    with pytest.raises(ValueError, match="together"):
        helpers.resolve_ssl(args)
    args = build_parser().parse_args(["--ssl-auto", "--ssl-keyfile", "key", "--ssl-certfile", "cert"])
    with pytest.raises(ValueError, match="combined"):
        helpers.resolve_ssl(args)
