"""Gradio ASR transcription and speech Qwen3 question-answering with sentence TTS.

ASR uses the original independent offline/causal Conformer checkpoints. QA uses
an acoustic encoder + projector + tuned Qwen3 and Qwen3-TTS CustomVoice. Record
or upload a complete utterance, then press Run: this is NOT continuous mic/VAD.
Text is incremental; audio is emitted as ordered, non-repeated sentence WAVs.
First audio waits for the first sentence AND its full TTS synthesis. Generation
can continue in the background during TTS; there is no codec-token streaming.

Run --help for all model/TLS options. Requires gradio>=4.44. Microphone access
requires localhost or trusted HTTPS. --ssl-auto creates a self-signed certificate
which requires browser approval (use --host with the actual address for its SAN).
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.speech_webui_helpers import (  # noqa: E402
    DEFAULT_INSTRUCTION, SAMPLE_RATE, SpeechEngine, decode_bytes_to_mono16k,
    resolve_ssl, stream_answer,
)

FRAME_MS, HOP_MS, N_MELS, BLANK = 25, 10, 80, 0
FPS = 1000 // HOP_MS
ASR_MODE = "ASR 转写（Conformer）"
QA_MODE = "语音问答 + 流式语音（Qwen3）"


def greedy_collapse(token_ids, id_to_char):
    """CTC collapse: blanks separate repeated labels."""
    out, previous = [], None
    for token in token_ids:
        if token == BLANK:
            previous = None
        elif token != previous:
            out.append(id_to_char[token])
            previous = token
    return "".join(out)


class OfflineASR:
    def __init__(self, checkpoint, device):
        import torch
        from model.ctc_model import TinyConformerCTC

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.id_to_char = {index: char for char, index in ck["vocab"].items()}
        self.model = TinyConformerCTC(len(ck["vocab"])).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def transcribe(self, features, device):
        import torch

        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x, None)
            length = self.model.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
            best = logits.argmax(dim=-1)[0, :int(length[0])].cpu().tolist()
        return greedy_collapse(best, self.id_to_char)


class StreamingASR:
    def __init__(self, checkpoint, chunk_size, left_context, device):
        import torch
        from model.ctc_streaming import TinyStreamingConformerCTC

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.id_to_char = {index: char for char, index in ck["vocab"].items()}
        self.model = TinyStreamingConformerCTC(
            len(ck["vocab"]), chunk_size=chunk_size, left_context=left_context,
        ).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.raw_per_chunk = chunk_size * 4

    def iter_incremental(self, features, device, cancelled=None):
        """Actual chunk inference, no timer/simulated playback or shared cache."""
        import torch

        n_raw = features.shape[0]
        feats = np.pad(features, ((0, (-n_raw) % self.raw_per_chunk), (0, 0)))
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
        encoder = self.model.encoder
        cache = encoder.init_cache(1, device, x.dtype)
        best = []
        for start in range(0, x.size(1), self.raw_per_chunk):
            if cancelled is not None and cancelled.is_set():
                return
            with torch.no_grad():
                out, cache = encoder.forward_chunk(x[:, start:start + self.raw_per_chunk], cache)
                logits = self.model.ctc_head(out)
            # Do not transcribe artificial padding in the last chunk.
            valid = min(logits.size(1), (min(self.raw_per_chunk, n_raw - start) + 3) // 4)
            best.extend(logits.argmax(dim=-1)[0, :valid].cpu().tolist())
            yield min(start + self.raw_per_chunk, n_raw) / FPS, greedy_collapse(best, self.id_to_char)

    def incremental(self, features, device):
        return list(self.iter_incremental(features, device))


class WebUIRunner:
    """Request-local outputs with cancellable, serialized shared-model access.

    Stop is cooperative rather than forcibly discarding a live generator: its
    finally block always closes/joins inference before releasing model ownership.
    Different browser sessions cannot stop each other's work.
    """
    def __init__(self, offline=None, streaming=None, engine=None, device="cpu", feature_fn=None):
        self.offline, self.streaming, self.engine = offline, streaming, engine
        self.device, self.feature_fn = device, feature_fn
        self._inference_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions = {}

    def cancel(self, session):
        with self._sessions_lock:
            event = self._sessions.get(session)
            if event is not None:
                event.set()
        return "已请求停止；正在运行的编码器 / TTS 调用结束后释放模型。"

    def run(self, path, mode, instruction, temperature, max_new_tokens, session):
        # (offline transcript, streaming transcript, answer, NEW WAV, status)
        offline_text, stream_text, answer = "", "", ""
        cancelled = threading.Event()
        acquired = False
        with self._sessions_lock:
            duplicate = session in self._sessions
            if not duplicate:
                self._sessions[session] = cancelled
        if duplicate:
            yield "", "", "", None, "该会话已有任务，请先停止或等待完成"
            return
        try:
            yield "", "", "", None, "等待模型…"
            if not path:
                raise ValueError("请先录音或上传音频")
            if mode not in (ASR_MODE, QA_MODE):
                raise ValueError("请选择有效模式")
            if mode == ASR_MODE and self.offline is None and self.streaming is None:
                raise ValueError("ASR 需要 --checkpoint 和/或 --stream-checkpoint")
            if mode == QA_MODE and (self.engine is None or not self.engine.tts_enabled):
                raise ValueError("语音问答需要 --projector-checkpoint、--qwen3-model 和 --tts-model")
            if not 1 <= int(max_new_tokens) <= 4096 or not 0 <= float(temperature) <= 2:
                raise ValueError("生成参数超出范围")
            while not cancelled.is_set():
                if self._inference_lock.acquire(timeout=0.1):
                    acquired = True
                    break
            if cancelled.is_set():
                yield "", "", "", None, "已停止"
                return
            audio = decode_bytes_to_mono16k(Path(path).read_bytes())
            if mode == ASR_MODE:
                if self.feature_fn is None:
                    from scripts.analyze_audio import log_mel
                    features, *_ = log_mel(audio, SAMPLE_RATE, FRAME_MS, HOP_MS, N_MELS)
                else:
                    features = self.feature_fn(audio)
                if self.streaming is not None:
                    for seconds, stream_text in self.streaming.iter_incremental(features, self.device, cancelled):
                        if cancelled.is_set():
                            break
                        yield "", stream_text, "", None, f"因果 ASR 已处理 {seconds:.2f}s"
                if self.offline is not None and not cancelled.is_set():
                    offline_text = self.offline.transcribe(features, self.device)
                yield offline_text, stream_text, "", None, "已停止" if cancelled.is_set() else "转写完成"
            else:
                source = stream_answer(self.engine, audio, instruction.strip() or DEFAULT_INSTRUCTION,
                                       float(temperature), int(max_new_tokens), cancelled)
                try:
                    for answer, wav, status in source:
                        yield "", "", answer, wav, status
                finally:
                    source.close()
        except Exception as exc:
            yield offline_text, stream_text, answer, None, f"错误：{exc}"
        finally:
            cancelled.set()
            if acquired:
                self._inference_lock.release()
            with self._sessions_lock:
                self._sessions.pop(session, None)


def build_demo(args, runner):
    import gradio as gr

    # A real Request annotation is needed by Gradio, despite future annotations.
    def run(path, mode, instruction, temperature, tokens, request):
        first = True
        source = runner.run(path, mode, instruction, temperature, tokens, request.session_hash)
        try:
            for row in source:
                # Text/status-only updates must not reset or replay the audio player.
                audio_chunk = row[3] if first or row[3] is not None else gr.skip()
                first = False
                yield row[0], row[1], row[2], audio_chunk, row[4]
        finally:
            source.close()

    def stop(request):
        return None, runner.cancel(request.session_hash)

    run.__annotations__["request"] = gr.Request
    stop.__annotations__["request"] = gr.Request
    with gr.Blocks(title="Speech-MiniMind · ASR / 语音问答") as demo:
        gr.Markdown(
            "# ASR 转写 / 语音问答\n"
            "录音结束或上传后点击开始。ASR 使用真实 Conformer CTC；问答使用语音 Qwen3，"
            "**不是把问答结果当作转写**。\n\n"
            "语音按句生成并依次播放：首包需等待首句文本及该句 TTS 合成，非音频 token 级流式。"
            "长句会分段；浏览器若限制自动播放，请手动点击播放器。停止不会中断正在运行的 TTS 内核。"
        )
        mode = gr.Radio([ASR_MODE, QA_MODE], value=QA_MODE if runner.engine else ASR_MODE, label="模式")
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                         value=str(args.audio) if args.audio else None, label="输入音频")
        instruction = gr.Textbox(value=args.instruction, label="问答系统指令（ASR 模式忽略）")
        with gr.Row():
            temperature = gr.Slider(0, 2, value=args.temperature, step=0.05, label="Temperature")
            tokens = gr.Slider(1, 4096, value=args.max_new_tokens, step=1, label="最大生成 token 数")
        with gr.Row():
            start = gr.Button("开始", variant="primary")
            cancel = gr.Button("停止")
        with gr.Row():
            offline = gr.Textbox(label="非流式 Conformer 转写", interactive=False)
            streaming = gr.Textbox(label="因果 Conformer 增量转写", interactive=False)
        answer = gr.Textbox(label="Qwen3 回答（增量文本）", interactive=False)
        speech = gr.Audio(label="回答语音（按句流式）", streaming=True, autoplay=True, format="wav")
        status = gr.Textbox(label="状态", interactive=False)
        start.click(run, [audio, mode, instruction, temperature, tokens],
                    [offline, streaming, answer, speech, status], concurrency_limit=None,
                    trigger_mode="once")
        # No forced generator cancellation: finish cleanup/join even after Stop.
        cancel.click(stop, outputs=[speech, status], queue=False)
    return demo


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="offline Conformer CTC checkpoint")
    parser.add_argument("--stream-checkpoint", type=Path, help="causal Conformer CTC checkpoint")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--left-context", type=int, default=16)
    parser.add_argument("--audio", type=Path, help="initial upload/example audio")
    parser.add_argument("--encoder-type", choices=("sensevoice", "conformer", "paraformer"), default="sensevoice")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--paraformer-model", default=None)
    parser.add_argument("--projector-checkpoint", type=Path)
    parser.add_argument("--qwen3-model", type=Path)
    parser.add_argument("--tune", choices=("lora", "full"), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--tts-model", type=Path, help="Qwen3-TTS CustomVoice directory")
    parser.add_argument("--tts-speaker", default="Serena")
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ssl-keyfile", type=Path)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-auto", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    qa_paths = (args.projector_checkpoint, args.qwen3_model, args.tts_model)
    if any(qa_paths) and not all(qa_paths):
        parser.error("QA requires --projector-checkpoint, --qwen3-model and --tts-model together")
    if not any(qa_paths) and not args.checkpoint and not args.stream_checkpoint:
        parser.error("provide ASR checkpoint(s) and/or the three QA model paths")
    if args.chunk_size < 1 or args.left_context < 0 or args.max_speech_tokens < 1:
        parser.error("chunk-size/max-speech-tokens must be positive; left-context must be nonnegative")
    if not 1 <= args.max_new_tokens <= 4096 or not 0 <= args.temperature <= 2:
        parser.error("max-new-tokens must be 1..4096; temperature must be 0..2")
    ssl = resolve_ssl(args)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline = OfflineASR(args.checkpoint, device) if args.checkpoint else None
    streaming = (StreamingASR(args.stream_checkpoint, args.chunk_size, args.left_context, device)
                 if args.stream_checkpoint else None)
    engine = None
    if all(qa_paths):
        from scripts.infer_speech_qwen3 import load_pipeline
        from qwen_tts import Qwen3TTSModel

        encoder, projector, lm, tokenizer, _ = load_pipeline(args, device)
        tts = Qwen3TTSModel.from_pretrained(
            str(args.tts_model), device_map=str(device),
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        engine = SpeechEngine(encoder, projector, lm, tokenizer, device, args.max_speech_tokens,
                              tts, args.tts_speaker, args.tts_language)
    runner = WebUIRunner(offline, streaming, engine, device)
    build_demo(args, runner).queue(max_size=32).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True, **ssl,
    )


if __name__ == "__main__":
    main()
