# Speech-MiniMind
从零开始、低成本、可复现的语音大模型项目，带学习者理解从音频波形到语音理解、语言推理和语音生成的完整过程。

## 学习路线

项目按可以独立运行的阶段逐步构建：

1. [语音基础](docs/00_audio_basics.md)：从 WAV、波形和频谱开始认识语音信号
2. 语音特征与小型声学编码器
3. CTC/Attention 语音识别
4. 音频表示接入 MiniMind 语言模型
5. 语音指令微调与语音问答
6. 流式推理与语音输出

## 快速开始：语音基础

```bash
python -m pip install -r requirements.txt
python scripts/analyze_audio.py examples/disgusted_to_happy.wav --plot outputs/example.png
python scripts/analyze_audio.py examples/disgusted_to_happy.wav \
  --stft-plot outputs/stft.png
python scripts/analyze_audio.py examples/disgusted_to_happy.wav \
  --stft-gif outputs/stft_process.gif
```

仓库自带的 `examples/disgusted_to_happy.wav` 是一个真实的 16-bit PCM WAV 示例。你也可以把命令中的路径替换为其他 WAV 文件。脚本会打印音频的采样率、时长、通道数、振幅范围、RMS 和过零率，并可生成波形与频谱图。音频文件不会被修改。

使用 `--stft-plot` 会把音频切成默认 25 ms 窗口、10 ms 帧移的多个片段，生成随时间变化的 STFT 时频图。
使用 `--stft-gif` 会生成动画：窗口沿波形移动，同时展示当前帧的频谱，并逐步累积 STFT 图。动画最多采样 240 个展示帧，避免 GIF 过大。
