# Speech-MiniMind
从零开始、低成本、可复现的语音大模型项目，带学习者理解从音频波形到语音理解、语言推理和语音生成的完整过程。

## 学习路线

项目按可以独立运行的阶段逐步构建：

1. [语音基础](docs/00_audio_basics.md)：从 WAV、波形和频谱开始认识语音信号
2. [Mel 频谱](docs/01_mel_spectrogram.md)：从 STFT 到 log-Mel 语音特征
3. [声学编码器与中文语音识别](docs/02_acoustic_encoder.md)：使用 Tiny Conformer + CTC 识别 AISHELL-1 中文语音
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
使用 `--stft-gif` 会生成动画：窗口沿波形移动，同时展示当前帧的频谱，并逐步累积 STFT 图。动画最多采样 120 个展示帧，并使用适合教学预览的分辨率。

## 02：训练中文语音识别模型

完整步骤请阅读 [02_acoustic_encoder.md](docs/02_acoustic_encoder.md)。最短可运行流程如下：

```bash
# 1. 下载 AISHELL-1 并解压
python scripts/download_aishell1.py

# 2. 生成 train/dev/test manifest 和字符词表
python scripts/prepare_aishell1.py

# 3. 训练 Tiny Conformer + CTC
python scripts/train_conformer_ctc.py --data data/aishell1/processed
```

AISHELL-1 压缩包约 15.6 GB，下载中断后直接重新运行下载命令即可断点续传。

训练完成后，checkpoint 位于 `data/aishell1/processed/tiny_conformer_ctc.pt`。当前模型为 4 层、hidden=256、4 头注意力、约 9.2M 参数；服务器训练时会自动使用 CUDA。

## 在 95 服务器训练

项目已经同步到：

```text
/gpu3/guhj/Speech-MiniMind
```

服务器环境已经配置为：

```bash
source /gpu/anaconda3/etc/profile.d/conda.sh
conda activate /gpu3/guhj/envs/speech-llm
cd /gpu3/guhj/Speech-MiniMind
```

然后执行上面的 AISHELL-1 下载、准备和训练命令即可。数据集和 checkpoint 默认保存在服务器项目目录下，不会被提交到 GitHub。
