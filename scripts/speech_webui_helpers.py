"""Reusable speech inference with incremental text and complete answer audio.

TTS synthesizes the full answer once after text generation finishes. Cancellation
stops generation between tokens; an in-flight encoder/TTS kernel must finish
before cleanup.
"""
from __future__ import annotations

import io
import queue
import re
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_INSTRUCTION = "你是一个语音助手，根据用户的音频内容回答用户的问题"


def decode_bytes_to_mono16k(data: bytes) -> np.ndarray:
    """Decode uploaded/recorded audio; ffmpeg or PCM16 WAV fallback."""
    if not data:
        raise ValueError("音频为空，请先录音或上传文件")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
                 "-ar", str(SAMPLE_RATE), "pipe:1"],
                input=data, capture_output=True, check=True, timeout=60,
            )
            audio = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("音频解码失败或超时，请检查文件") from exc
    else:
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                if wav.getsampwidth() != 2:
                    raise ValueError("未安装 ffmpeg 时只支持 PCM16 WAV")
                rate, channels = wav.getframerate(), wav.getnchannels()
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                audio = audio.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
                if rate != SAMPLE_RATE and audio.size:
                    audio = np.interp(np.arange(round(audio.size * SAMPLE_RATE / rate)) / SAMPLE_RATE,
                                      np.arange(audio.size) / rate, audio).astype(np.float32)
        except (wave.Error, EOFError) as exc:
            raise ValueError("无法解码音频；非 PCM16 WAV 需要安装 ffmpeg") from exc
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("音频为空或包含无效采样")
    return audio


def split_sentence_chunks(text: str, max_chars: int = 100, final: bool = False):
    """Commit punctuation-terminated/bounded chunks, retaining the unspoken tail."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks = []
    punctuation = re.compile(r"[。！？!?；;：:]|[.](?=\s|$)")
    while text:
        match = punctuation.search(text, 0, max_chars)
        if match:
            end = match.end()
        elif len(text) > max_chars:
            end = text.rfind(" ", 0, max_chars + 1)
            if end <= 0:
                end = max_chars
        elif final:
            end = len(text)
        else:
            break
        chunk, text = text[:end].strip(), text[end:]
        if chunk:
            chunks.append(chunk)
    return chunks, text


def stream_generated_text(model, tokenizer, inputs_embeds, attention_mask,
                          max_new_tokens, temperature, cancelled):
    """Join the owned generation thread on every exit, propagating its errors.

    Unlike the CLI's streamer, early generator close must not leave a GPU worker
    behind while another request acquires the model lock.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    stopped = threading.Event()

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancelled.is_set() or stopped.is_set()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                    timeout=0.1)
    errors = []
    done = threading.Event()

    def generate():
        try:
            with torch.no_grad():
                model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=bool(getattr(model.config, "use_cache", True)),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                    do_sample=temperature > 0,
                    **({"temperature": temperature, "top_p": 0.95, "top_k": 20}
                       if temperature > 0 else {}),
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([Cancelled()]),
                )
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=generate, name="speech-webui-generate", daemon=True)
    worker.start()
    text = ""
    try:
        while not cancelled.is_set():
            try:
                piece = next(streamer)
            except queue.Empty:
                if done.is_set():
                    break
                continue
            except StopIteration:
                break
            text += piece
            yield text
    finally:
        stopped.set()
        worker.join()  # Never release model ownership while generation still runs.
    if errors:
        raise RuntimeError(f"Qwen3 生成失败: {errors[0]}") from errors[0]


class SpeechEngine:
    """Shared models; serialize requests, keep all text/audio state request-local."""

    def __init__(self, encoder, projector, lm, tokenizer, device, max_speech_tokens,
                 tts_model=None, tts_speaker="Serena", tts_language="Chinese"):
        self.encoder, self.projector = encoder, projector
        self.lm, self.tokenizer, self.device = lm, tokenizer, device
        self.max_speech_tokens = max_speech_tokens
        self.tts_model, self.tts_speaker, self.tts_language = tts_model, tts_speaker, tts_language
        self._lock = threading.Lock()
        self._tts_lock = threading.Lock()

    @property
    def tts_enabled(self):
        return self.tts_model is not None

    def stream(self, audio, instruction, temperature, max_new_tokens, cancelled=None):
        import torch
        from model.qwen3_adapter import _prompt_embeds

        if not audio.size:
            raise ValueError("音频为空")
        with self._lock:
            with torch.no_grad():
                acoustic, lengths = self.encoder.encode(
                    torch.from_numpy(audio[None].astype(np.float32)),
                    torch.tensor([audio.size], dtype=torch.long), SAMPLE_RATE,
                )
                projected = self.projector(acoustic)
                lengths = self.projector.output_lengths(lengths).clamp_max(projected.size(1))
                embeds, mask = _prompt_embeds(
                    self.lm, self.tokenizer, projected, lengths, instruction, self.device,
                    self.max_speech_tokens,
                )
            yield from stream_generated_text(self.lm, self.tokenizer, embeds, mask,
                                            max_new_tokens, temperature,
                                            cancelled if cancelled is not None else threading.Event())

    @staticmethod
    def _tts_text(text):
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"https?://\S+", "网址", text)
        text = re.sub(r"[*_>#~]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:4000]

    def synthesize(self, text):
        """Return one PCM16 WAV, keeping original amplitude (no per-chunk scaling)."""
        if not self.tts_enabled:
            raise RuntimeError("请通过 --tts-model 加载 Qwen3-TTS")
        clean = self._tts_text(text)
        if not clean:
            raise ValueError("没有可合成的文本")
        with self._tts_lock:
            wavs, rate = self.tts_model.generate_custom_voice(
                text=clean, language=self.tts_language, speaker=self.tts_speaker,
            )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if not audio.size or not np.isfinite(audio).all() or int(rate) <= 0:
            raise ValueError("TTS 返回无效音频")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(rate))
            wav.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        return output.getvalue(), int(rate)


def stream_answer(engine, audio, instruction, temperature, max_new_tokens, cancelled=None):
    """Stream text, then synthesize and emit the complete answer audio once.

    Close/join text generation before TTS. Cancellation discards in-flight audio;
    synthesis failures preserve the answer without retrying.
    """
    if not engine.tts_enabled:
        raise ValueError("语音问答模式需要 --tts-model（Qwen3-TTS CustomVoice）")
    previous = ""
    cancelled = cancelled if cancelled is not None else threading.Event()
    source = engine.stream(audio, instruction, temperature, max_new_tokens, cancelled)
    try:
        for text in source:
            if cancelled.is_set():
                break
            previous = text
            yield text, None, "生成回答中…"
    finally:
        source.close()
    if cancelled.is_set():
        yield previous, None, "已停止"
        return
    if not previous.strip():
        yield previous, None, "模型返回空回答"
        return
    yield previous, None, "正在合成完整语音，请稍候…"
    if cancelled.is_set():
        yield previous, None, "已停止"
        return
    try:
        wav, _ = engine.synthesize(previous)
    except Exception as exc:
        status = "已停止" if cancelled.is_set() else f"语音合成失败（文本保留）: {exc}"
        yield previous, None, status
    else:
        if cancelled.is_set():
            yield previous, None, "已停止"
        else:
            yield previous, wav, "完成"


def resolve_ssl(args):
    """Gradio TLS kwargs; --ssl-auto persists localhost/listen-host certificates."""
    if args.ssl_auto and (args.ssl_keyfile or args.ssl_certfile):
        raise ValueError("--ssl-auto cannot be combined with explicit TLS files")
    if bool(args.ssl_keyfile) != bool(args.ssl_certfile):
        raise ValueError("--ssl-keyfile and --ssl-certfile must be provided together")
    if args.ssl_auto:
        import ipaddress
        directory = Path(__file__).resolve().parent / ".webui_ssl"
        key, cert = directory / "key.pem", directory / "cert.pem"
        if not key.exists() or not cert.exists():
            directory.mkdir(parents=True, exist_ok=True)
            names = ["DNS:localhost", "IP:127.0.0.1", "IP:::1"]
            if args.host not in ("0.0.0.0", "::", "localhost"):
                try:
                    ipaddress.ip_address(args.host)
                    names.append(f"IP:{args.host}")
                except ValueError:
                    names.append(f"DNS:{args.host}")
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-batch",
                "-keyout", str(key), "-out", str(cert), "-days", "365",
                "-subj", "/CN=localhost", "-addext", "subjectAltName=" + ",".join(names),
            ], check=True, capture_output=True)
            key.chmod(0o600)
        return {"ssl_keyfile": str(key), "ssl_certfile": str(cert), "ssl_verify": False}
    if args.ssl_keyfile:
        return {"ssl_keyfile": str(args.ssl_keyfile), "ssl_certfile": str(args.ssl_certfile)}
    return {}
