"""Gradio web UI to test the instruction-tuned Speech-MiniMind (chapter 04).

Loads the same frozen-encoder + projector + tuned-MiniMind pipeline as
``infer_speech_minimind.py`` and exposes it as an interactive playground:
upload a WAV, pick a text instruction (or type one), and see the model's
answer. A waveform of the input is drawn for context.

Usage
-----
.. code-block:: bash

    pip install gradio

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

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse the shared load pipeline of the CLI so behaviour stays identical
from scripts.infer_speech_minimind import load_pipeline, resample_to_16k  # noqa: E402
from model.minimind_adapter import stream_from_speech  # noqa: E402
from scripts.infer_speech_minimind import SAMPLE_RATE  # noqa: E402

PRESET_INSTRUCTIONS = [
    "请将这段语音准确转写为中文文本。",
    "简要概括这段语音说了什么。",
    "这段语音属于什么话题？请用一句话回答。",
    "请把这段语音翻译成英文。",
    "请提取语音中的关键信息（时间、人物、地点）。",
]


def build_waveform(audio: np.ndarray, sample_rate: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(audio.size) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=110)
    ax.plot(time, audio, linewidth=0.4, color="0.25")
    ax.set(xlim=(time[0], time[-1]) if audio.size else (0, 1),
           xlabel="time (s)", ylabel="amp")
    ax.set_title(f"input audio — {audio.size / sample_rate:.1f}s")
    fig.tight_layout()
    return fig


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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio is required. Run: python -m pip install gradio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    encoder, projector, lm, tokenizer, _ = load_pipeline(args, device)
    print("pipeline ready — serving UI")

    def do_run(audio_input: tuple | list | None, instruction: str):
        # gr.Audio(type="numpy") returns (sample_rate, waveform) where waveform
        # is float32 in [-1,1] with shape (N,) or (N, channels). Generator fn so
        # the answer streams into the UI token-by-token instead of all at once.
        if not audio_input:
            raise gr.Error("请先上传或录制一段语音")
        rate, audio = audio_input
        instruction = (instruction or "").strip()
        if not instruction:
            raise gr.Error("请输入或选择一条指令")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # downmix multi-channel
        try:
            audio, rate = resample_to_16k(audio, rate)
        except Exception as ex:
            raise gr.Error(f"音频解码/重采样失败: {ex}")
        fig = build_waveform(audio, rate)
        yield fig, "⏳ 正在编码语音…"

        waveforms = torch.from_numpy(audio[None].astype(np.float32))
        lengths = torch.tensor([audio.size], dtype=torch.long)
        with torch.no_grad():
            acoustic, acoustic_lengths = encoder.encode(waveforms, lengths, SAMPLE_RATE)
            projected = projector(acoustic)
            projected_lengths = projector.output_lengths(acoustic_lengths).clamp_max(projected.size(1))

        yield fig, "⏳ 模型生成中…"
        for partial, _complete in stream_from_speech(
            lm, tokenizer, projected, projected_lengths, instruction, device,
            max_new_tokens=args.max_new_tokens,
            max_speech_tokens=args.max_speech_tokens,
            temperature=args.temperature,
        ):
            yield fig, partial if partial else "…"

    with gr.Blocks(title="Speech-MiniMind · 指令微调模型测试平台") as demo:
        gr.Markdown(
            "# 指令微调 Speech-MiniMind 测试平台\n\n"
            "上传或录制一段**中文语音**，输入（或选择）一条指令，模型会基于语音内容生成回答。"
            "任意格式的音频会自动解码并重采样到 16kHz（`paraformer` 前端要求）。"
        )
        with gr.Row():
            audio_upload = gr.Audio(type="numpy", label="上传 / 录制语音")
        instruction_input = gr.Dropdown(
            choices=PRESET_INSTRUCTIONS, value=PRESET_INSTRUCTIONS[0],
            label="指令（点击选择预设，或手动输入）", allow_custom_value=True,
        )
        run_btn = gr.Button("▶ 运行", variant="primary")

        fig = gr.Plot(label="输入波形")
        answer_out = gr.Markdown(label="模型回答")

        run_btn.click(
            fn=do_run,
            inputs=[audio_upload, instruction_input],
            outputs=[fig, answer_out],
        )

    demo.queue().launch(server_name=args.host, server_port=args.port,
                        show_error=True, share=False)
    print(f"gradio serving at http://{args.host}:{args.port}")


if __name__ == "__main__":
    main()