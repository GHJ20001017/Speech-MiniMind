# Speech-MiniMind

### 从一段 WAV 开始，亲手构建一个中文 Speech LLM

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Dataset](https://img.shields.io/badge/Dataset-AISHELL--1-2ea44f)](https://www.openslr.org/33/)

中文 · [English](#english)

Speech-MiniMind 是一个面向初学者的 Speech LLM 学习项目。它不从封装好的 ASR 接口开始，而是展示从采样点、FFT、Mel 三角滤波器到 Conformer 的每个中间步骤。

目标是：能看懂、能运行、能修改。当前重点是中文声学编码器和字符级 CTC 语音识别，不直接追求商用 ASR 指标。

## 学习路线

~~~text
00 语音基础 → 01 Mel 频谱 → 02 Tiny Conformer + CTC
                                      ↓
                         03 语音表示接入 MiniMind
                                      ↓
                         04 流式语音理解与输出
~~~

| 章节 | 内容 | 状态 |
|---|---|---|
| [00. 语音基础](docs/00_audio_basics.md) | WAV、波形、FFT、频谱泄漏、STFT | 已完成 |
| [01. Mel 频谱](docs/01_mel_spectrogram.md) | 功率谱、Mel 滤波器组、log-Mel | 已完成 |
| [02. 声学编码器](docs/02_acoustic_encoder.md) | Tiny Conformer、AISHELL-1、CTC | 训练中 |
| 03. 语音接入 MiniMind | 声学 token 与语言模型连接 | 计划中 |
| 04. 流式语音理解 | 实时推理和语音输出 | 计划中 |

## 当前完成内容

- WAV、PCM、采样率、振幅、RMS 和过零率分析；
- 分帧、Hann 窗、FFT、功率谱、STFT 动画；
- 80 维 log-Mel 特征提取；
- 4 层 Tiny Conformer 中文声学编码器；
- AISHELL-1 字符级 CTC 训练；
- train/dev loss、loss 曲线和逐 epoch checkpoint；
- ModelScope 国内镜像、变长 batch 和 padding mask。

## 快速开始

~~~bash
conda create -n speech-llm python=3.11
conda activate speech-llm
python -m pip install -r requirements.txt
~~~

使用仓库中的示例音频生成波形、频谱和 STFT 动画：

~~~bash
python scripts/analyze_audio.py examples/disgusted_to_happy.wav \
  --plot outputs/example.png \
  --stft-plot outputs/stft.png \
  --stft-gif outputs/stft_process.gif
~~~

## 训练中文语音识别

### 1. 获取 AISHELL-1

AISHELL-1 压缩包约 15.6 GB，国内用户建议使用 [ModelScope 镜像](https://www.modelscope.cn/datasets/OmniData/AISHELL-1/tree/master/raw/33)：

~~~bash
python scripts/download_aishell1.py
~~~

脚本支持断点续传。也可以网页下载 data_aishell.tgz 后放到 data/aishell1/，再运行上面的命令自动解压。

### 2. 生成 manifest 和词表

~~~bash
python scripts/prepare_aishell1.py
~~~

输出：

~~~text
data/aishell1/processed/
├── train.csv
├── dev.csv
├── test.csv
└── vocab.txt
~~~

### 3. 开始训练

~~~bash
python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4
~~~

训练过程使用 tqdm，并保存：

~~~text
outputs/02_acoustic_encoder/
├── config.json
├── metrics.csv
├── loss_curve.png
├── checkpoint_epoch_001.pt
└── tiny_conformer_ctc.pt
~~~

data/、outputs/、.pt 和压缩包已加入 .gitignore，不会上传到 GitHub。

## 模型配置

| 项目 | 配置 |
|---|---:|
| 输入 | 80 维 log-Mel |
| Conformer 层数 | 4 |
| hidden dimension | 256 |
| 注意力头数 | 4 |
| FFN dimension | 1024 |
| 卷积 kernel | 31 |
| 时间下采样 | 4 倍 |
| 参数量 | 约 9.0M，随词表大小变化 |
| 训练目标 | 字符级 CTC |

完整公式、张量形状、CTC 对齐和常见问题见 [02_acoustic_encoder.md](docs/02_acoustic_encoder.md)。

## 在 95 服务器训练

~~~bash
source /gpu/anaconda3/etc/profile.d/conda.sh
conda activate /gpu3/guhj/envs/speech-llm
cd /gpu3/guhj/Speech-MiniMind

CUDA_VISIBLE_DEVICES=7 python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4
~~~

后台训练并保存终端日志：

~~~bash
mkdir -p outputs/02_acoustic_encoder
nohup env CUDA_VISIBLE_DEVICES=7 \
  python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  > outputs/02_acoustic_encoder/train.log 2>&1 &
~~~

查看进度：

~~~bash
tail -f outputs/02_acoustic_encoder/train.log
nvidia-smi
~~~

## 训练完成后

当前训练脚本记录 CTC loss。完整评估还需要 CTC 解码和 CER：

~~~text
checkpoint → log-Mel → Conformer + CTC
          → greedy CTC decode → 中文文本 → CER
~~~

下一步将加入评估脚本，输出 dev/test CER，并保存参考文本与预测文本样例。

## 目录结构

~~~text
Speech-MiniMind/
├── docs/                   # 分章节教学文档
├── examples/               # 示例音频
├── model/                  # Conformer 和 CTC 模型
├── scripts/                # 分析、准备和训练脚本
├── data/                   # 本地数据，不提交
├── outputs/                # 图表、日志、checkpoint，不提交
├── requirements.txt
└── README.md
~~~

## 开源说明

项目代码处于教学开发阶段。数据集遵循 AISHELL-1 原始许可和使用条款；数据文件、日志、checkpoint 和个人音频不应提交到仓库。正式发布前会补充明确的代码许可证和数据集引用说明。

## English

Speech-MiniMind is an educational, from-scratch Speech LLM project. It explains the path from waveform and FFT to log-Mel features, a compact Conformer encoder, and Chinese character-level CTC speech recognition on AISHELL-1.

~~~text
WAV → STFT → log-Mel → Tiny Conformer → CTC → Chinese characters
~~~

See [docs/00_audio_basics.md](docs/00_audio_basics.md), [docs/01_mel_spectrogram.md](docs/01_mel_spectrogram.md), and [docs/02_acoustic_encoder.md](docs/02_acoustic_encoder.md) for the tutorials. Dataset files, logs, checkpoints, and generated figures are ignored by Git.
