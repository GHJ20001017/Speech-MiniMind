"""Web UI to test the instruction-tuned Speech-MiniMind (chapter 04).

Two interaction modes over one page, both backed by the same frozen-encoder +
projector + tuned-MiniMind pipeline as ``infer_speech_minimind.py``:

* **audio upload**  — pick/drop a WAV (or any format ffmpeg can read), the
  server decodes it to 16 kHz mono, runs the model and streams the answer.
* **microphone**    — the browser captures 16 kHz mono PCM and keeps a single
  WebSocket open. A server-side energy **VAD** watches the stream, and as soon
  as it detects the end of an utterance it runs the model, streaming the text
  answer back token-by-token. No button press per sentence: talk, pause, read.

The microphone needs a secure context (``https`` or ``localhost``); the
recommended setup is an SSH/port-forward so the page is served on
``http://localhost:<port>``.

Usage
-----
.. code-block:: bash

    pip install fastapi uvicorn

    # full-mode tuned model, paraformer frontend (recommended)
    python scripts/visualize_speech_minimind_webui.py \\
      --encoder-type paraformer \\
      --paraformer-model outputs/paraformer-streaming \\
      --projector-checkpoint outputs/03_speech_minimind_paraformer/projector_epoch_005.pt \\
      --minimind-model outputs/04_speech_minimind_sft/model_epoch_003 \\
      --host 0.0.0.0 --port 7861

All weights come from the paths you pass; nothing is downloaded here.
Open http://<host>:<port> in a browser to use it.
"""

import argparse
import asyncio
import io
import json
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse the shared load pipeline of the CLI so behaviour stays identical
from scripts.infer_speech_minimind import load_pipeline, SAMPLE_RATE  # noqa: E402
from model.minimind_adapter import stream_from_speech  # noqa: E402

PRESET_INSTRUCTIONS = [
    "请将这段语音准确转写为中文文本。",
    "简要概括这段语音说了什么。",
    "这段语音属于什么话题？请用一句话回答。",
    "请把这段语音翻译成英文。",
    "请提取语音中的关键信息（时间、人物、地点）。",
]


# --------------------------------------------------------------------------- #
# VAD: energy-based, adaptive noise floor, no external dependency.
# --------------------------------------------------------------------------- #
class EnergyVAD:
    """Frame-level energy VAD with an adaptive noise floor.

    Feed fixed-size frames through :meth:`process`. It returns ``("start", None)``
    once an utterance begins, ``("end", audio)`` with the collected waveform when
    the utterance ends (by trailing silence or a hard duration cap), else
    ``None``. The noise floor tracks the ambient level during silence so the
    decision thresholds adapt to the room instead of relying on a magic number.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: float = 30.0,
        start_mult: float = 2.5,
        stop_mult: float = 1.5,
        min_speech_ms: float = 250.0,
        silence_ms: float = 700.0,
        max_utterance_s: float = 30.0,
        noise_floor_init: float = 1e-4,
        noise_floor_max: float = 0.05,
        noise_adapt: float = 0.05,
        adaptive: bool = True,
        trim_tail: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
        self.start_mult = start_mult
        self.stop_mult = stop_mult
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.silence_frames_needed = max(1, round(silence_ms / frame_ms))
        self.max_frames = max(1, round(max_utterance_s * 1000.0 / frame_ms))
        self.noise_floor = noise_floor_init
        self.noise_floor_max = noise_floor_max
        self.noise_adapt = noise_adapt
        self.adaptive = adaptive
        self.trim_tail = trim_tail
        self._leftover = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Clear the utterance state (keeps the learned noise floor)."""
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.buffer: list[np.ndarray] = []
        self._leftover = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame)) + 1e-12))

    def _thresholds(self) -> tuple[float, float]:
        start = max(self.noise_floor * self.start_mult, self.noise_floor + 1e-4)
        stop = self.noise_floor * self.stop_mult
        return start, stop

    def _advance(self, frame: np.ndarray) -> tuple[str, np.ndarray | None] | None:
        rms = self._rms(frame)
        start_thr, stop_thr = self._thresholds()

        if not self.in_speech:
            if rms > start_thr:
                self.speech_frames += 1
                self.buffer.append(frame)
                if self.speech_frames >= self.min_speech_frames:
                    self.in_speech = True
                    self.silence_frames = 0
                    return ("start", None)
            else:
                self.speech_frames = 0
                self.buffer = []
                if self.adaptive:  # learn the room noise while idle
                    self.noise_floor = min(
                        self.noise_floor_max,
                        max(1e-5, (1.0 - self.noise_adapt) * self.noise_floor
                            + self.noise_adapt * rms),
                    )
            return None

        self.buffer.append(frame)
        if rms < stop_thr:
            self.silence_frames += 1
        else:
            self.silence_frames = 0

        if self.silence_frames >= self.silence_frames_needed or len(self.buffer) >= self.max_frames:
            frames = self.buffer
            if self.trim_tail:  # drop most of the trailing silence we waited for
                frames = frames[: max(1, len(frames) - self.silence_frames_needed + 2)]
            audio = (
                np.concatenate(frames).astype(np.float32)
                if frames else np.zeros(0, dtype=np.float32)
            )
            self.in_speech = False
            self.speech_frames = 0
            self.silence_frames = 0
            self.buffer = []
            return ("end", audio)
        return None

    def process(self, chunk: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        """Feed an arbitrary-length float32 chunk; return the events it produced."""
        events: list[tuple[str, np.ndarray | None]] = []
        data = np.concatenate([self._leftover, chunk.astype(np.float32)])
        n_frames = data.size // self.frame_len
        for i in range(n_frames):
            frame = data[i * self.frame_len:(i + 1) * self.frame_len]
            ev = self._advance(frame)
            if ev is not None:
                events.append(ev)
        self._leftover = data[n_frames * self.frame_len:]
        return events


# --------------------------------------------------------------------------- #
# Audio decoding for the upload mode.
# --------------------------------------------------------------------------- #
def decode_bytes_to_mono16k(data: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes to a mono float32 waveform at 16 kHz.

    Prefers ffmpeg (broad format support); falls back to the stdlib ``wave``
    reader for plain 16-bit PCM WAV when ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
                input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
            audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
            if audio.size:
                return audio
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="ignore").strip().splitlines()
            raise ValueError(f"ffmpeg 解码失败: {detail[-1] if detail else exc}") from exc

    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except Exception as exc:  # noqa: BLE001
        raise ValueError("无法解码音频（系统未找到 ffmpeg，且不是 16-bit PCM WAV）") from exc

    if width != 2:
        raise ValueError(f"回退解码只支持 16-bit PCM WAV，实际 {width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and audio.size:
        new_len = round(audio.size * SAMPLE_RATE / rate)
        audio = np.interp(
            np.arange(new_len) / SAMPLE_RATE, np.arange(audio.size) / rate, audio
        ).astype(np.float32)
    return audio.astype(np.float32)


# --------------------------------------------------------------------------- #
# Inference engine: shared, serialized, streams partial text.
# --------------------------------------------------------------------------- #
class SpeechEngine:
    """Frozen encoder + projector + tuned MiniMind behind a single lock."""

    def __init__(self, encoder, projector, lm, tokenizer, device, max_speech_tokens: int) -> None:
        self.encoder = encoder
        self.projector = projector
        self.lm = lm
        self.tokenizer = tokenizer
        self.device = device
        self.max_speech_tokens = max_speech_tokens
        self._lock = threading.Lock()

    def stream(self, audio: np.ndarray, instruction: str,
               temperature: float, max_new_tokens: int):
        """Yield ``(partial_text, complete_text)`` for one utterance."""
        with self._lock:  # one GPU inference at a time
            if audio.size == 0:
                return
            waveforms = torch.from_numpy(audio[None].astype(np.float32))
            lengths = torch.tensor([audio.size], dtype=torch.long)
            with torch.no_grad():
                acoustic, acoustic_lengths = self.encoder.encode(waveforms, lengths, SAMPLE_RATE)
                projected = self.projector(acoustic)
                projected_lengths = self.projector.output_lengths(acoustic_lengths).clamp_max(
                    projected.size(1)
                )
            yield from stream_from_speech(
                self.lm, self.tokenizer, projected, projected_lengths, instruction, self.device,
                max_new_tokens=max_new_tokens,
                max_speech_tokens=self.max_speech_tokens,
                temperature=temperature,
            )


# --------------------------------------------------------------------------- #
# Per-connection session state.
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, engine: SpeechEngine, vad_kwargs: dict, defaults: dict) -> None:
        self.engine = engine
        self.vad_kwargs = dict(vad_kwargs)
        self.instruction = defaults["instruction"]
        self.temperature = defaults["temperature"]
        self.max_new_tokens = defaults["max_new_tokens"]
        self.vad = EnergyVAD(**self.vad_kwargs)
        self.mode: str | None = None           # "mic" | "upload" | None
        self.upload_expected = 0
        self.upload_chunks: list[bytes] = []

    def _apply_vad(self, payload: dict) -> None:
        for key in ("start_mult", "stop_mult", "silence_ms", "min_speech_ms"):
            if key in payload:
                self.vad_kwargs[key] = float(payload[key])
        self.vad = EnergyVAD(**self.vad_kwargs)

    async def handle_text(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "无法解析的消息"})
            return
        kind = msg.get("type")
        if kind == "set_instruction":
            self.instruction = (msg.get("text") or "").strip() or self.instruction
        elif kind == "set_params":
            if "temperature" in msg:
                self.temperature = float(msg["temperature"])
            if "max_new_tokens" in msg:
                self.max_new_tokens = int(msg["max_new_tokens"])
        elif kind == "set_vad":
            self._apply_vad(msg)
            await ws.send_json({"type": "info", "message": "VAD 参数已更新"})
        elif kind == "mic_start":
            self.mode = "mic"
            self.vad.reset()
            await ws.send_json({"type": "status", "value": "listening"})
        elif kind == "mic_stop":
            self.mode = None
            self.vad.reset()
            await ws.send_json({"type": "status", "value": "idle"})
        elif kind == "upload_begin":
            self.mode = "upload"
            self.upload_expected = int(msg.get("size") or 0)
            self.upload_chunks = []
            if msg.get("instruction"):
                self.instruction = msg["instruction"].strip()
            await ws.send_json({"type": "info", "message": "开始接收音频…"})
        elif kind == "cancel":
            self.mode = None
            self.upload_chunks = self.upload_chunks[:0]
            self.vad.reset()
            await ws.send_json({"type": "status", "value": "idle"})
        else:
            await ws.send_json({"type": "error", "message": f"未知消息类型: {kind}"})

    async def handle_binary(self, ws, data: bytes) -> None:
        if self.mode == "upload":
            self.upload_chunks.append(data)
            received = sum(len(c) for c in self.upload_chunks)
            if self.upload_expected and received < self.upload_expected:
                return
            blob = b"".join(self.upload_chunks)
            self.mode = None
            self.upload_chunks = []
            await ws.send_json({"type": "status", "value": "decoding"})
            try:
                audio = decode_bytes_to_mono16k(blob)
            except ValueError as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                await ws.send_json({"type": "status", "value": "idle"})
                return
            await ws.send_json({"type": "audio", "duration": round(audio.size / SAMPLE_RATE, 2)})
            await self._run(ws, audio)
        elif self.mode == "mic":
            pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            for event, audio in self.vad.process(pcm):
                if event == "start":
                    await ws.send_json({"type": "status", "value": "speech"})
                elif event == "end" and audio is not None and audio.size:
                    await ws.send_json({"type": "audio",
                                        "duration": round(audio.size / SAMPLE_RATE, 2)})
                    await self._run(ws, audio)
                    if self.mode == "mic":
                        await ws.send_json({"type": "status", "value": "listening"})

    async def _run(self, ws, audio: np.ndarray) -> None:
        """Run one utterance and stream the answer back to the client."""
        await ws.send_json({"type": "status", "value": "generating"})
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                for partial, complete in self.engine.stream(
                    audio, self.instruction, self.temperature, self.max_new_tokens
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("partial", complete))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", ""))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        full = ""
        while True:
            kind, payload = await queue.get()
            if kind == "partial":
                full = payload
                await ws.send_json({"type": "partial", "text": payload})
            elif kind == "error":
                await ws.send_json({"type": "error", "message": payload})
                break
            else:
                break
        await ws.send_json({"type": "final", "text": full})


# --------------------------------------------------------------------------- #
# HTML / JS single page.
# --------------------------------------------------------------------------- #
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Speech-MiniMind · 测试平台</title>
<style>
  :root { --fg:#1f2328; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --bg:#f7f7f8; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif; }
  header { padding:20px 28px; border-bottom:1px solid var(--line); background:#fff; }
  header h1 { margin:0 0 4px; font-size:20px; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .wrap { max-width:960px; margin:0 auto; padding:20px 28px 60px; }
  .tabs { display:flex; gap:6px; margin-bottom:18px; }
  .tab { padding:8px 16px; border:1px solid var(--line); border-radius:8px; background:#fff;
         cursor:pointer; font-size:14px; }
  .tab.active { border-color:var(--accent); color:var(--accent); font-weight:600; }
  .panel { display:none; background:#fff; border:1px solid var(--line); border-radius:12px;
           padding:20px; }
  .panel.active { display:block; }
  label { display:block; font-size:13px; color:var(--muted); margin:12px 0 6px; }
  input[type=text], select { width:100%; padding:9px 11px; border:1px solid var(--line);
         border-radius:8px; font:inherit; background:#fff; }
  button { font:inherit; padding:9px 18px; border-radius:8px; border:1px solid var(--accent);
           background:var(--accent); color:#fff; cursor:pointer; }
  button.ghost { background:#fff; color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .status { display:inline-flex; align-items:center; gap:8px; font-size:13px; color:var(--muted); }
  .dot { width:10px; height:10px; border-radius:50%; background:#cbd5e1; }
  .dot.listening { background:#22c55e; } .dot.speech { background:#f59e0b; }
  .dot.generating { background:var(--accent); animation:pulse 1s infinite; }
  .dot.idle { background:#cbd5e1; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  canvas { width:100%; height:70px; border:1px solid var(--line); border-radius:8px; background:#fafafa; }
  .answer { margin-top:16px; padding:14px 16px; border:1px solid var(--line); border-radius:8px;
            background:#fcfcfd; min-height:52px; white-space:pre-wrap; }
  .answer .placeholder { color:#9ca3af; }
  .err { color:#b91c1c; }
  .hint { font-size:12px; color:var(--muted); margin-top:6px; }
  .sliders { display:grid; grid-template-columns:1fr 1fr; gap:0 20px; }
  input[type=range] { width:100%; }
</style>
</head>
<body>
<header>
  <h1>指令微调 Speech-MiniMind 测试平台</h1>
  <p>同一个模型，两种用法：上传一段音频，或打开麦克风连续说话（服务端 VAD 自动断句）。</p>
</header>
<div class="wrap">
  <div class="tabs">
    <div class="tab active" data-tab="upload">① 音频上传</div>
    <div class="tab" data-tab="mic">② 麦克风实时</div>
  </div>

  <!-- upload -->
  <div class="panel active" id="panel-upload">
    <label>选择音频文件（WAV / MP3 / M4A 等，服务端自动解码并重采样到 16 kHz）</label>
    <input type="file" id="file" accept="audio/*"/>
    <label>指令</label>
    <select id="instruction-upload"></select>
    <div class="row" style="margin-top:14px">
      <button id="run-upload">▶ 运行</button>
      <span class="status"><span class="dot" id="up-dot"></span><span id="up-status">就绪</span></span>
    </div>
    <label>输入波形</label>
    <canvas id="up-canvas" width="900" height="140"></canvas>
    <div class="answer" id="up-answer"><span class="placeholder">模型回答将显示在这里</span></div>
  </div>

  <!-- mic -->
  <div class="panel" id="panel-mic">
    <p class="hint">点击「开始监听」后保持 WebSocket 常连：对着麦克风说话，停顿约 0.7 秒后模型自动开始识别并流式输出文本。</p>
    <label>指令</label>
    <select id="instruction-mic"></select>
    <div class="row" style="margin-top:14px">
      <button id="mic-toggle">🎙 开始监听</button>
      <span class="status"><span class="dot idle" id="mic-dot"></span><span id="mic-status">未连接</span></span>
      <span class="status">WS: <span id="ws-state">连接中…</span></span>
    </div>
    <label>麦克风电平</label>
    <canvas id="mic-canvas" width="900" height="140"></canvas>
    <div class="sliders">
      <div>
        <label>VAD 灵敏度（越小越灵敏）<span id="sens-val"></span></label>
        <input type="range" id="sens" min="1.3" max="6" step="0.1"/>
      </div>
      <div>
        <label>断句静音时长 <span id="sil-val"></span></label>
        <input type="range" id="sil" min="300" max="2000" step="50"/>
      </div>
    </div>
    <div class="answer" id="mic-answer"><span class="placeholder">麦克风识别的文本将显示在这里</span></div>
  </div>
</div>

<script>
const PRESETS = __PRESETS__;
const $ = (id) => document.getElementById(id);

function fillSelect(sel) {
  sel.innerHTML = "";
  for (const p of PRESETS) {
    const o = document.createElement("option"); o.value = p; o.textContent = p; sel.appendChild(o);
  }
  const custom = document.createElement("option");
  custom.value = "__custom__"; custom.textContent = "（手动输入…）"; sel.appendChild(custom);
}
fillSelect($("instruction-upload"));
fillSelect($("instruction-mic"));

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $("panel-" + t.dataset.tab).classList.add("active");
}));

function currentInstruction(sel) {
  if (sel.value === "__custom__") {
    const v = prompt("请输入指令：", PRESETS[0]);
    if (v) { const o = document.createElement("option"); o.value = v; o.textContent = v;
             sel.insertBefore(o, sel.lastElementChild); sel.value = v; }
    else sel.value = PRESETS[0];
  }
  return sel.value;
}

// ---- shared websocket -----------------------------------------------------
let ws = null, wsReady = false;
const pending = [];              // queue messages until the socket is open
function wsSend(obj) {
  const s = JSON.stringify(obj);
  if (wsReady) ws.send(s); else pending.push(s);
}
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsReady = true; $("ws-state").textContent = "已连接";
                      while (pending.length) ws.send(pending.shift()); };
  ws.onclose = () => { wsReady = false; $("ws-state").textContent = "已断开，重连中…";
                       setTimeout(connectWS, 1500); };
  ws.onerror = () => { $("ws-state").textContent = "连接错误"; };
}
connectWS();

// ---- upload mode ----------------------------------------------------------
function drawWaveform(canvas, samples, color) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fafafa"; ctx.fillRect(0, 0, W, H);
  if (!samples || !samples.length) return;
  const step = Math.max(1, Math.floor(samples.length / W));
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 0; x < W; x++) {
    let min = 1, max = -1;
    for (let i = x * step; i < (x + 1) * step && i < samples.length; i++) {
      min = Math.min(min, samples[i]); max = Math.max(max, samples[i]);
    }
    ctx.moveTo(x, (1 - max) * H / 2);
    ctx.lineTo(x, (1 - min) * H / 2);
  }
  ctx.stroke();
}

$("run-upload").addEventListener("click", async () => {
  const f = $("file").files[0];
  if (!f) { alert("请先选择一个音频文件"); return; }
  $("run-upload").disabled = true;
  setStatus("up-dot", "generating", "处理中…");
  $("up-answer").innerHTML = '<span class="placeholder">⏳ 正在解码并生成…</span>';
  const buf = await f.arrayBuffer();
  drawWaveformFromBlob(buf);
  uploadTarget = "up";
  wsSend({ type: "set_instruction", text: currentInstruction($("instruction-upload")) });
  wsSend({ type: "upload_begin", name: f.name, size: buf.byteLength });
  if (!wsReady) await new Promise((r) => { const t = setInterval(() => { if (wsReady) { clearInterval(t); r(); } }, 100); });
  ws.send(buf);
});
function drawWaveformFromBlob(buf) {
  try {
    const tmp = new (window.AudioContext || window.webkitAudioContext)();
    tmp.decodeAudioData(buf.slice(0)).then((ab) => {
      drawWaveform($("up-canvas"), ab.getChannelData(0), "#2563eb");
      tmp.close();
    }).catch(() => {});
  } catch (e) { /* ignore */ }
}

// ---- microphone mode ------------------------------------------------------
let audioCtx = null, micStream = null, procNode = null, muteNode = null;
let resampler = null, sending = false, micOn = false;

class Resampler {
  constructor(inRate, outRate) { this.ratio = inRate / outRate; this.buf = new Float32Array(0); this.pos = 0; }
  process(input) {
    if (this.ratio === 1) return input;
    const merged = new Float32Array(this.buf.length + input.length);
    merged.set(this.buf); merged.set(input, this.buf.length);
    const outLen = Math.max(0, Math.floor((merged.length - 1 - this.pos) / this.ratio));
    const out = new Float32Array(outLen);
    let pos = this.pos;
    for (let i = 0; i < outLen; i++) {
      const i0 = Math.floor(pos), frac = pos - i0;
      const a = merged[i0], b = merged[i0 + 1];
      out[i] = a + (b - a) * frac;
      pos += this.ratio;
    }
    const keepFrom = Math.floor(pos);
    this.buf = merged.slice(keepFrom);
    this.pos = pos - keepFrom;
    return out;
  }
}
function floatTo16(f) {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

$("mic-toggle").addEventListener("click", async () => {
  if (!micOn) await startMic(); else stopMic();
});

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (e) { alert("无法访问麦克风：" + e.message + "\n（需 localhost 或 https）"); return; }
  try { audioCtx = new AudioContext({ sampleRate: 16000 }); }
  catch (e) { audioCtx = new AudioContext(); }
  await audioCtx.resume();
  resampler = new Resampler(audioCtx.sampleRate, 16000);
  const src = audioCtx.createMediaStreamSource(micStream);
  procNode = audioCtx.createScriptProcessor(1024, 1, 1);
  muteNode = audioCtx.createGain(); muteNode.gain.value = 0;   // avoid local playback
  procNode.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    drawWaveform($("mic-canvas"), input, "#16a34a");
    if (!sending) return;
    const res = resampler.process(new Float32Array(input));
    if (!res.length) return;
    if (wsReady) ws.send(floatTo16(res).buffer);
  };
  src.connect(procNode); procNode.connect(muteNode); muteNode.connect(audioCtx.destination);

  micOn = true; sending = true;
  $("mic-toggle").textContent = "■ 停止监听";
  setStatus("mic-dot", "listening", "监听中…");
  wsSend({ type: "set_instruction", text: currentInstruction($("instruction-mic")) });
  wsSend({ type: "set_vad", start_mult: parseFloat($("sens").value),
           silence_ms: parseFloat($("sil").value) });
  wsSend({ type: "mic_start" });
  $("mic-answer").innerHTML = '<span class="placeholder">正在聆听…</span>';
}

function stopMic() {
  micOn = false; sending = false;
  wsSend({ type: "mic_stop" });
  if (procNode) { procNode.disconnect(); procNode.onaudioprocess = null; }
  if (muteNode) muteNode.disconnect();
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  $("mic-toggle").textContent = "🎙 开始监听";
  setStatus("mic-dot", "idle", "已停止");
}

$("sens").addEventListener("input", () => {
  $("sens-val").textContent = " (" + $("sens").value + ")";
  if (micOn) wsSend({ type: "set_vad", start_mult: parseFloat($("sens").value) });
});
$("sil").addEventListener("input", () => {
  $("sil-val").textContent = " (" + $("sil").value + " ms)";
  if (micOn) wsSend({ type: "set_vad", silence_ms: parseFloat($("sil").value) });
});
$("sens-val").textContent = " (" + $("sens").value + ")";
$("sil-val").textContent = " (" + $("sil").value + " ms)";

// ---- incoming messages ----------------------------------------------------
let uploadTarget = null;
function setStatus(dotId, cls, text) {
  const d = $(dotId); d.className = "dot " + cls;
  $(dotId === "up-dot" ? "up-status" : "mic-status").textContent = text;
}
function answerEl() { return $("mic-answer"); }   // mic is the streaming target

ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.type === "status") {
    const isUpload = m.value === "decoding" || (uploadTarget === "up" && m.value !== "listening");
    if (uploadTarget === "up") {
      if (m.value === "generating") setStatus("up-dot", "generating", "生成中…");
    } else {
      setStatus("mic-dot", m.value, { listening: "监听中…", speech: "检测到语音…",
        generating: "生成中…", idle: micOn ? "监听中…" : "已停止", decoding: "处理中…" }[m.value] || m.value);
    }
  } else if (m.type === "audio") {
    const label = "🎧 检测到语音（" + m.duration + "s），生成中…";
    if (uploadTarget === "up") $("up-answer").innerHTML = '<span class="placeholder">' + label + '</span>';
    else $("mic-answer").innerHTML = '<span class="placeholder">' + label + '</span>';
  } else if (m.type === "partial") {
    if (uploadTarget === "up") $("up-answer").textContent = m.text;
    else $("mic-answer").textContent = m.text;
  } else if (m.type === "final") {
    if (uploadTarget === "up") {
      $("up-answer").textContent = m.text || "（空回答）";
      setStatus("up-dot", "idle", "完成");
      $("run-upload").disabled = false;
      uploadTarget = null;
    } else {
      $("mic-answer").textContent = m.text || "（空回答）";
    }
  } else if (m.type === "error") {
    if (uploadTarget === "up") { $("up-answer").innerHTML = '<span class="err">' + m.message + '</span>';
      setStatus("up-dot", "idle", "出错"); $("run-upload").disabled = false; uploadTarget = null; }
    else { $("mic-answer").innerHTML = '<span class="err">' + m.message + '</span>';
      setStatus("mic-dot", "idle", "出错"); }
  } else if (m.type === "info") {
    // informational, ignore
  }
};
</script>
</body>
</html>
"""


def build_app(engine: SpeechEngine, vad_kwargs: dict, defaults: dict, host: str, port: int):
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    app = FastAPI(title="Speech-MiniMind WebUI")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE_HTML.replace("__PRESETS__", json.dumps(PRESET_INSTRUCTIONS, ensure_ascii=False))

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.post("/api/infer")
    async def infer(request: Request):
        """One-shot HTTP inference for scripts/tests: raw audio bytes -> JSON answer."""
        instruction = request.query_params.get("instruction", defaults["instruction"])
        data = await request.body()
        if not data:
            return JSONResponse({"error": "empty body"}, status_code=400)
        try:
            audio = decode_bytes_to_mono16k(data)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        text = ""
        for _partial, complete in engine.stream(
            audio, instruction, defaults["temperature"], defaults["max_new_tokens"]
        ):
            text = complete
        return JSONResponse({"answer": text, "duration": round(audio.size / SAMPLE_RATE, 2)})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = Session(engine, vad_kwargs, defaults)
        await ws.send_json({"type": "ready", "ws": True})
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await session.handle_binary(ws, bytes(message["bytes"]))
                elif message.get("text") is not None:
                    await session.handle_text(ws, message["text"])
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:  # noqa: BLE001
                pass

    return app, uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-type", choices=("conformer", "paraformer"), default="paraformer",
                        help="frozen acoustic encoder backend")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--paraformer-model", default=None)
    parser.add_argument("--projector-checkpoint", type=Path, required=True)
    parser.add_argument("--minimind-model", type=Path, required=True,
                        help="chapter-04 tuned MiniMind dir (full or lora)")
    parser.add_argument("--tune", choices=("lora", "full"), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--instruction", default=PRESET_INSTRUCTIONS[0],
                        help="default instruction for both modes")
    # VAD knobs (energy-based)
    parser.add_argument("--vad-frame-ms", type=float, default=30.0)
    parser.add_argument("--vad-start-mult", type=float, default=2.5,
                        help="speech starts when frame RMS > noise_floor * this")
    parser.add_argument("--vad-stop-mult", type=float, default=1.5,
                        help="speech ends when frame RMS < noise_floor * this")
    parser.add_argument("--vad-min-speech-ms", type=float, default=250.0)
    parser.add_argument("--vad-silence-ms", type=float, default=700.0,
                        help="trailing silence that ends an utterance")
    parser.add_argument("--vad-max-utterance-s", type=float, default=30.0)
    parser.add_argument("--no-vad-adaptive", action="store_true",
                        help="freeze the noise floor instead of tracking ambient level")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit("fastapi/uvicorn are required. Run: python -m pip install fastapi uvicorn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    encoder, projector, lm, tokenizer, _ = load_pipeline(args, device)
    engine = SpeechEngine(encoder, projector, lm, tokenizer, device, args.max_speech_tokens)

    vad_kwargs = dict(
        sample_rate=SAMPLE_RATE,
        frame_ms=args.vad_frame_ms,
        start_mult=args.vad_start_mult,
        stop_mult=args.vad_stop_mult,
        min_speech_ms=args.vad_min_speech_ms,
        silence_ms=args.vad_silence_ms,
        max_utterance_s=args.vad_max_utterance_s,
        adaptive=not args.no_vad_adaptive,
    )
    defaults = dict(
        instruction=args.instruction,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    print("pipeline ready — serving UI")
    app, uvicorn = build_app(engine, vad_kwargs, defaults, args.host, args.port)
    print(f"serving at http://{args.host}:{args.port}  (open http://localhost:{args.port} for mic)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
