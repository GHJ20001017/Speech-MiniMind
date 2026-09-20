"""Dependency-free tests for sentence streaming helpers and embedded browser JS."""
import ast
import asyncio
import base64
import pathlib
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "visualize_speech_qwen3_webui.py"


def source_tree():
    return ast.parse(SOURCE.read_text())


def load_splitter():
    node = next(n for n in source_tree().body if isinstance(n, ast.FunctionDef)
                and n.name == "split_sentence_chunks")
    ns = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["split_sentence_chunks"]


def load_run():
    cls = next(n for n in source_tree().body if isinstance(n, ast.ClassDef) and n.name == "Session")
    node = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run")
    ns = {"asyncio": asyncio, "base64": base64, "queue": queue,
          "threading": threading, "np": types.SimpleNamespace(ndarray=object),
          "split_sentence_chunks": load_splitter()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["_run"]


def test_splitter_order_and_tail():
    split = load_splitter()
    chunks, tail = split("第一句。第二句")
    assert chunks == ["第一句。"] and tail == "第二句"
    chunks, tail = split(tail, final=True)
    assert chunks == ["第二句"] and tail == ""


def test_splitter_bounded_fallback_and_empty():
    split = load_splitter()
    chunks, tail = split("x" * 205, max_chars=100)
    assert [len(x) for x in chunks] == [100, 100] and len(tail) == 5
    assert split("   ", final=True) == ([], "")


def test_async_streams_before_completion_and_preserves_tail_on_tts_failure():
    run = load_run()
    released = threading.Event()
    calls = []

    class Engine:
        tts_enabled = True
        @staticmethod
        def _tts_text(text): return text.strip()
        def stream(self, *_args):
            yield "", "第一句。"
            assert released.wait(2)
            yield "", "第一句。第二句"
        def synthesize(self, text):
            calls.append(text)
            if text == "第二句":
                raise RuntimeError("synthetic failure")
            return b"wav", 16000

    class WS:
        def __init__(self): self.messages = []; self.first_audio = threading.Event()
        async def send_json(self, msg):
            self.messages.append(msg)
            if msg.get("type") == "speech_audio": self.first_audio.set()

    obj = types.SimpleNamespace(engine=Engine(), instruction="", temperature=0,
                                max_new_tokens=8, _runs=set(), _cancel_queues=set())
    ws = WS()
    async def exercise():
        task = asyncio.create_task(run(obj, ws, None))
        await asyncio.to_thread(ws.first_audio.wait, 2)
        assert calls == ["第一句。"]  # first audio precedes LLM completion
        released.set()
        await task
    asyncio.run(exercise())
    assert calls == ["第一句。", "第二句"]
    assert [m["type"] for m in ws.messages].count("tts_error") == 1
    assert ws.messages[-1] == {"type": "final", "text": "第一句。第二句"}


def test_cancellation_guards_and_browser_queue_regressions():
    source = SOURCE.read_text()
    assert "if cancelled.is_set() or loop.is_closed()" in source
    assert "self._cancel_queues" in source and "q.put(None)" in source
    assert "speechQueue.unshift(item)" in source
    assert "generation !== speechGeneration" in source
    assert 'm.type === "tts_error"' in source


def test_node_audio_queue_behavior():
    node = shutil.which("node")
    if not node:
        return
    source = SOURCE.read_text()
    html = source.split('PAGE_HTML = r"""', 1)[1].split('"""', 1)[0]
    html = html.replace("\\\\u003c", "<").replace("\\\\u003e", ">")
    block = "let speechQueue" + html.split("let speechQueue", 1)[1].split("function handleMessage", 1)[0]
    harness = r'''const assert = require("assert");
const window = global.window = {};
const button = {hidden: true, addEventListener(_kind, fn) { this.fn = fn; }};
function $(id) { assert.equal(id, "speech-enable"); return button; }
let urls = [], revoked = [], rejectPlay = false, made = [];
global.URL = {createObjectURL() { const u = "u" + urls.length; urls.push(u); return u; }, revokeObjectURL(u) { revoked.push(u); }};
global.Blob = class { constructor() {} };
global.atob = x => x;
global.Audio = class { constructor(url) { this.url=url; made.push(this); }
  play() { return rejectPlay ? Promise.reject(new Error("blocked")) : Promise.resolve(); }
  pause() {}
};
''' + block + r'''
enqueueSpeechAudio("a"); assert.equal(made.length, 1);
const staleEnded = made[0].onended; clearSpeechQueue(); staleEnded();
assert.equal(made.length, 1); assert(revoked.length >= 1);
rejectPlay = true; enqueueSpeechAudio("b");
Promise.resolve().then(() => Promise.resolve()).then(() => {
  assert.equal(button.hidden, false); rejectPlay = false; button.fn();
  return Promise.resolve();
}).then(() => { assert.equal(made.length, 3); console.log("node audio queue passed"); });
'''
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness); path = f.name
    try:
        subprocess.run([node, path], check=True)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_embedded_javascript_syntax():
    node = shutil.which("node")
    if not node:
        return
    source = SOURCE.read_text()
    html = source.split('PAGE_HTML = r"""', 1)[1].split('"""', 1)[0]
    html = html.replace("\\u003c", "<").replace("\\u003e", ">")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script); path = f.name
    try:
        subprocess.run([node, "--check", path], check=True)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_splitter_order_and_tail()
    test_splitter_bounded_fallback_and_empty()
    test_async_streams_before_completion_and_preserves_tail_on_tts_failure()
    test_cancellation_guards_and_browser_queue_regressions()
    test_node_audio_queue_behavior()
    test_embedded_javascript_syntax()
    print("streaming TTS tests passed")
