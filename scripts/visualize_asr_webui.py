"""Gradio web UI that visualises the chapter-02 ASR encoder, streaming vs offline.

It loads the *non-streaming* Tiny Conformer CTC model (:class:`model.ctc_model
.TinyConformerCTC`) and the *streaming* causal Conformer CTC model
(:class:`model.ctc_streaming.TinyStreamingConformerCTC`), then lets you watch a
long utterance be recognised two ways side by side:

* **Left — non-streaming, "pseudo-streaming" (伪流式)**: a non-causal model has
  no stable incremental output, so to force text out before the audio ends we
  re-run it on the *growing prefix* each time.  Text does appear incrementally,
  but it is expensive (O(N^2)) and **already-emitted words can be revised** as
  more audio arrives.
* **Right — streaming (增量)**: audio is fed chunk by chunk (``chunk_size``
  block-frames, left-context cache carried between chunks); recognised text
  appears *incrementally* and is **causal/stable** — once a word is out it is
  never rewritten.
* **Bottom — non-streaming, whole utterance (整句)**: the model's real usage;
  the full transcript only appears once the whole utterance has been read in.

Seeing the left text flip while the right stays put is exactly the difference
between a non-causal and a causal acoustic encoder.

A wave plot + progress slider + optional auto-play simulate the streaming
read-in, so the difference is visible without a microphone.

Usage
-----
.. code-block:: bash

    pip install gradio

    python scripts/visualize_asr_webui.py \\
      --checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \\
      --stream-checkpoint outputs/02_streaming_acoustic_encoder/checkpoint_epoch_001.pt \\
      --audio path/to/long.wav

    # offline side only (streaming side disabled)
    python scripts/visualize_asr_webui.py --checkpoint ... --audio path/to/long.wav

    # streaming side only (offline side disabled)
    python scripts/visualize_asr_webui.py --stream-checkpoint ... --audio path/to/long.wav

Each side is optional: omitting a checkpoint shows that side as "unavailable".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.ctc_model import TinyConformerCTC  # noqa: E402
from model.ctc_streaming import TinyStreamingConformerCTC  # noqa: E402
from scripts.analyze_audio import log_mel, read_wav  # noqa: E402

FRAME_MS = 25
HOP_MS = 10
N_MELS = 80
BLANK = 0
FPS = 1000 // HOP_MS  # log-mel frames per second (always 100 for HOP_MS=10)


def greedy_collapse(token_ids: list[int] | np.ndarray, id_to_char: dict[int, str]) -> str:
    """CTC collapse: drop blanks, fold adjacent duplicates -> text string."""
    out: list[str] = []
    prev: int | None = None
    for token_id in token_ids:
        if token_id == BLANK:
            prev = None
            continue
        if token_id == prev:
            continue
        out.append(id_to_char[token_id])
        prev = token_id
    return "".join(out)


class OfflineASR:
    """Whole-utterance (non-streaming) recogniser."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        vocab = ck["vocab"]
        self.id_to_char = {index: char for char, index in vocab.items()}
        self.model = TinyConformerCTC(len(vocab)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def transcribe(self, features: np.ndarray, device: torch.device) -> str:
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x, None)
            out_len = self.model.encoder.subsampled_lengths(lengths).clamp_max(logits.size(1))
            best = logits.argmax(dim=-1)[0, : int(out_len[0])].cpu().tolist()
        return greedy_collapse(best, self.id_to_char)

    def incremental_prefix(
        self, features: np.ndarray, device: torch.device, n_points: int = 20
    ) -> list[tuple[float, str]]:
        """Simulate streaming output from a NON-causal model.

        The non-streaming encoder attends over the whole utterance, so it has
        no stable incremental output.  The only faithful way to get text out of
        it before the audio ends is to re-run the model on the *growing prefix*
        each time.  Returns ``[(end_seconds_k, text_on_prefix_k), ...]``.

        This is deliberately labelled "pseudo-streaming" in the UI: it is
        O(N^2) expensive and, because the model still sees future frames inside
        each prefix, already-emitted text can be revised as more audio arrives.
        """
        n_raw = features.shape[0]
        if n_raw == 0:
            return []
        # keep at least ~1s of audio per point (the 4x subsampling needs a few
        # frames) and cap the number of re-runs so the pre-compute stays fast
        max_points = max(2, n_raw // FPS)
        n_points = max(2, min(n_points, max_points))
        grid = sorted({int(round(k * n_raw / n_points)) for k in range(1, n_points + 1)})
        grid[-1] = n_raw
        milestones: list[tuple[float, str]] = []
        for end in grid:
            milestones.append((end / FPS, self.transcribe(features[:end], device)))
        return milestones


class StreamingASR:
    """Chunk-wise (incremental) recogniser.

    We pre-compute every chunk's collapsed text together with its cumulative
    end-time offset, so the UI can scrub the progress bar freely without
    re-running the model on each position (keeps the animation smooth). The
    incremental property is preserved: text shown at time ``t`` is exactly the
    CTC collapse of all frames fed up to that time.
    """

    def __init__(self, checkpoint: Path, chunk_size: int, left_context: int, device: torch.device) -> None:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        vocab = ck["vocab"]
        self.id_to_char = {index: char for char, index in vocab.items()}
        self.model = TinyStreamingConformerCTC(
            len(vocab), chunk_size=chunk_size, left_context=left_context
        ).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.raw_per_chunk = chunk_size * 4  # 4x time subsampling

    def incremental(self, features: np.ndarray, device: torch.device) -> list[tuple[float, str]]:
        """Return ``[(end_seconds_k, text_after_chunk_k), ...]`` over the audio.

        The final entry's time equals the true audio duration (padding clamped).
        """
        n_raw = features.shape[0]
        pad = (-n_raw) % self.raw_per_chunk
        feats = np.pad(features, ((0, pad), (0, 0)))
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
        encoder = self.model.encoder
        cache = encoder.init_cache(1, device, x.dtype)
        cum_best: list[int] = []
        milestones: list[tuple[float, str]] = []
        seconds_per_chunk = self.raw_per_chunk / FPS
        for start in range(0, x.size(1), self.raw_per_chunk):
            piece = x[:, start : start + self.raw_per_chunk]
            with torch.no_grad():
                out, cache = encoder.forward_chunk(piece, cache)
                logits = self.model.ctc_head(out)
            cum_best.extend(logits.argmax(dim=-1)[0].cpu().tolist())
            text = greedy_collapse(cum_best, self.id_to_char)
            n_done = (start // self.raw_per_chunk + 1) * self.raw_per_chunk
            milestones.append((min(n_done, n_raw) / FPS, text))
        return milestones


def build_waveform(audio: np.ndarray, sample_rate: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(audio.size) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=110)
    ax.plot(time, audio, linewidth=0.4, color="0.25")
    ax.set(xlim=(time[0], time[-1]), xlabel="time (s)", ylabel="amp")
    ax.set_title(f"input audio — {audio.size / sample_rate:.1f}s")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="non-streaming CTC checkpoint")
    parser.add_argument(
        "--stream-checkpoint",
        type=Path,
        default=None,
        help="streaming CTC checkpoint (independent of --checkpoint; omit to disable the streaming side)",
    )
    parser.add_argument("--audio", type=Path, default=None, help="a 16-bit PCM wav path shown in the dropdown")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--left-context", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:  # pragma: no cover
        raise SystemExit("gradio is required. Run: python -m pip install gradio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    offline, streaming = None, None
    if args.checkpoint is not None:
        offline = OfflineASR(args.checkpoint, device)
        print(f"loaded non-streaming: {args.checkpoint}")
    # The offline and streaming encoders have different subsampling and conv
    # layouts, so their checkpoints are NOT interchangeable. Only enable the
    # streaming side when an explicit streaming checkpoint is given; never
    # fall back to --checkpoint (that would try to load offline weights into
    # the streaming model and fail with a state_dict mismatch).
    if args.stream_checkpoint is not None:
        streaming = StreamingASR(args.stream_checkpoint, args.chunk_size, args.left_context, device)
        print(f"loaded streaming:      {args.stream_checkpoint}")
    else:
        print("loaded streaming:      (none — pass --stream-checkpoint to enable)")

    # shared decoded state, filled on first run
    state = {
        "dur_s": 0.0,
        "offline_text": None,
        "offline_prefix": None,
        "milestones": None,
        "fig": None,
    }

    def reset() -> None:
        state.update(
            dur_s=0.0, offline_text=None, offline_prefix=None, milestones=None, fig=None
        )

    def do_run(audio_path: str) -> tuple:
        reset()
        if not audio_path:
            raise gr.Error("请先上传或选择音频文件")
        audio, rate = read_wav(Path(audio_path))
        features, *_ = log_mel(audio, rate, FRAME_MS, HOP_MS, N_MELS)
        state["dur_s"] = audio.size / rate
        state["fig"] = build_waveform(audio, rate)
        if offline is not None:
            # true non-streaming output (needs the whole utterance) ...
            state["offline_text"] = offline.transcribe(features, device)
            # ... plus the growing-prefix re-runs that fake streaming output
            state["offline_prefix"] = offline.incremental_prefix(features, device)
        if streaming is not None:
            state["milestones"] = streaming.incremental(features, device)
        # reset playback to 0 and set the slider max to the new duration
        return (
            state["fig"],
            gr.update(maximum=state["dur_s"], value=0.0),
            offline_prefix_view(0.0),
            streaming_view(0.0),
            offline_final_view(0.0),
        )

    def offline_prefix_view(pos: float) -> str:
        """Pseudo-streaming: non-streaming model re-run on the growing prefix."""
        if offline is None:
            return "（非流式模型未加载：未提供 --checkpoint）"
        prefix = state.get("offline_prefix")
        if prefix is None:
            return "⏳ 识别中…"
        text = ""
        for end_s, t in prefix:
            if end_s <= pos + 1e-3:
                text = t
        if not text:
            text = "…尚无输出…"
        return (
            f"**{text}**\n\n"
            f"（已输入 {pos:.1f}s / {state['dur_s']:.1f}s）\n\n"
            "> 非流式模型没有因果性，只能把“到目前为止的整段音频”反复重跑。"
            "后面的音频一到，**已经出现的文字可能被改写**。"
        )

    def offline_final_view(pos: float) -> str:
        if offline is None:
            return "（非流式模型未加载：未提供 --checkpoint）"
        if state["offline_text"] is None:
            return "⏳ 识别中…"
        dur = state["dur_s"]
        if pos < dur - 1e-3:
            return f"⏳ 音频尚未读完（{pos:.1f}s / {dur:.1f}s）\n\n整句识别需要等**全部音频输入完成**后才出结果。"
        return f"非流式（整句）最终结果：\n\n**{state['offline_text']}**"

    def streaming_view(pos: float) -> str:
        if streaming is None:
            return "（流式模型未加载：未提供 --stream-checkpoint）"
        milestones = state.get("milestones")
        if milestones is None:
            return "⏳ 识别中…"
        text = ""
        for end_s, t in milestones:
            if end_s <= pos + 1e-3:
                text = t
        if not text:
            text = "…正在听，暂无输出…"
        return (
            f"**{text}**\n\n"
            f"（已输入 {pos:.1f}s / {state['dur_s']:.1f}s）\n\n"
            "> 流式模型是因果的：每个字的输出只依赖已输入的音频，"
            "**一旦出现就不会被后续音频改写**。"
        )

    def on_slider(pos: float) -> tuple:
        return (offline_prefix_view(pos), streaming_view(pos), offline_final_view(pos))

    with gr.Blocks(title="Speech-MiniMind · 第 02 章 ASR 编码器可视化") as demo:
        gr.Markdown(
            "# 声学编码器 ASR 可视化（流式 vs 非流式）\n\n"
            "**左「非流式 · 伪流式」**：把非流式模型在“增长前缀”上反复重跑，"
            "强行挤出的增量输出——会随音频增长被**修正**。\n\n"
            "**右「流式 · 真流式」**：因果模型，文本随音频**边输入边生成**，"
            "出现即稳定。\n\n"
            "**底部「非流式 · 整句」**：非流式模型真正的用法，只有整段音频输完才出最终结果。"
        )
        with gr.Row():
            audio_upload = gr.Audio(type="filepath", label="上传 / 选择 WAV（模型在 16kHz 中文朗读上训练，AISHELL 语料为佳）")
            run_btn = gr.Button("▶ 开始识别", variant="primary")
        audio_pick = gr.Dropdown(
            choices=[str(args.audio)] if args.audio else [],
            label="或选择预设长音频",
            visible=bool(args.audio),
        )

        fig = gr.Plot(label="输入波形")
        slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="音频进度 (s)", interactive=True)
        play_toggle = gr.Checkbox(label="⏵ 自动播放（模拟实时读入）")
        play_state = gr.State(False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 左 — 非流式模型 · 伪流式（增长前缀重跑）")
                offline_prefix_out = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### 右 — 流式模型 · 真流式（增量、稳定）")
                streaming_out = gr.Markdown()
        gr.Markdown("### 非流式模型 · 整句（真正的非流式用法）")
        offline_final_out = gr.Markdown()

        if args.audio:

            def use_preset(chosen):
                return chosen if chosen else None

            run_btn.click(
                fn=use_preset,
                inputs=[audio_pick],
                outputs=[audio_upload],
            ).then(
                fn=do_run,
                inputs=[audio_upload],
                outputs=[fig, slider, offline_prefix_out, streaming_out, offline_final_out],
            )
        else:
            run_btn.click(
                fn=do_run,
                inputs=[audio_upload],
                outputs=[fig, slider, offline_prefix_out, streaming_out, offline_final_out],
            )

        slider.release(
            fn=on_slider,
            inputs=[slider],
            outputs=[offline_prefix_out, streaming_out, offline_final_out],
        )
        play_toggle.change(
            fn=lambda on: on,
            inputs=[play_toggle],
            outputs=[play_state],
        )
        timer = gr.Timer(value=0.2)

        def tick(play_on: bool, pos: float | None) -> tuple:
            # NOTE: values must arrive via `inputs`; reading the component
            # object's ``.value`` would only ever see the initial state.
            pos = pos if pos is not None else 0.0
            if play_on and state["dur_s"] > 0:
                pos = min(pos + state["dur_s"] / 120.0, state["dur_s"])
            return pos, offline_prefix_view(pos), streaming_view(pos), offline_final_view(pos)

        timer.tick(
            fn=tick,
            inputs=[play_state, slider],
            outputs=[slider, offline_prefix_out, streaming_out, offline_final_out],
        )

    demo.queue().launch(server_name=args.host, server_port=args.port, show_error=True, share=False)
    print(f"gradio serving at http://{args.host}:{args.port}")


if __name__ == "__main__":
    main()