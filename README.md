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
python scripts/analyze_audio.py path/to/example.wav --plot outputs/example.png
```

脚本会打印音频的采样率、时长、通道数、振幅范围、RMS 和过零率，并可生成波形与频谱图。音频文件不会被修改。
