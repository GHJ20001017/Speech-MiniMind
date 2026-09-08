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
| [03. 语音接入 MiniMind](docs/03_speech_minimind.md) | Speech Projector、语音前缀、MiniMind | 实现中 |
| 04. 流式语音理解 | 实时推理和语音输出 | 计划中 |

## 当前完成内容

- WAV、PCM、采样率、振幅、RMS 和过零率分析；
- 分帧、Hann 窗、FFT、功率谱、STFT 动画；
- 80 维 log-Mel 特征提取；
- 4 层 Tiny Conformer 中文声学编码器；
- AISHELL-1 字符级 CTC 训练；
- train/dev loss、loss 曲线和逐 epoch checkpoint；
- ModelScope 国内镜像、变长 batch 和 padding mask。
- Speech Projector 的最小桥接训练脚本（将继续完善推理和指令数据）。

## 03. 将声学编码器接入 MiniMind

02 中的 Conformer 输出是 `[语音帧数, 256]`，而 MiniMind 的词向量维度通常是 768，不能直接拼接。03 增加一个可训练的 `SpeechProjector`：先用一维卷积将约 10 ms 一帧的声学序列降采样，再用 MLP 映射到 MiniMind hidden size。

```text
log-Mel [T, 80] → Tiny Conformer [T/4, 256]
                         ↓ SpeechProjector
                    speech prefix [T/16, 768]
                         ↓ 拼到文本 embedding 前
                    MiniMind → 中文文本
```

### 准备 MiniMind 模型

MiniMind 源码和权重不复制进本仓库。请从 [MiniMind 官方仓库](https://github.com/jingyaogong/minimind) 的模型链接下载 Transformers 格式权重，例如 [minimind-3](https://huggingface.co/jingyaogong/minimind-3)，保存到本地目录。目录中应包含 `config.json`、tokenizer 文件和模型权重。

### 运行最小桥接训练

先确认 02 的 CTC checkpoint 已存在，并安装 `transformers`：

```bash
python -m pip install transformers
python scripts/train_speech_minimind.py \
  --data data/aishell1/processed \
  --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \
  --minimind-model /path/to/minimind-3 \
  --output outputs/03_speech_minimind \
  --epochs 3 \
  --batch-size 2
```

该版本默认冻结 Tiny Conformer 和 MiniMind，只训练约 0.8M 参数的 Projector。AISHELL-1 仍然只提供“语音→文字”监督，因此这是语音接入语言模型的教学桥接实验，不是完整的语音问答训练。训练日志会写入 `outputs/03_speech_minimind/metrics.csv`，Projector checkpoint 写入同目录。

### 构造第二阶段语音指令数据

可以把 AISHELL-1 的转写标注转换为统一的语音指令格式：

```bash
python scripts/prepare_speech_instructions.py \
  --input data/aishell1/processed \
  --output data/speech_instructions
```

每行是一个 JSON 对象：

```json
{"audio":"...wav","instruction":"请将这段语音准确转写为中文文本。","answer":"今天天气很好。","task":"transcription"}
```

如果需要少量可验证的指令跟随样例，可以额外打开 `--include-text-ops`，它会加入“转写后统计汉字数、找首字、找末字”任务。它们只使用已有转写文本推导答案，不伪造音频中没有的信息；真正的语音摘要、问答和意图识别仍需要额外的人工标注或公开语音指令数据。

### 构建小规模第二阶段混合数据

为了控制小模型的训练成本，可以用混合构建脚本先生成约 5,000 条数据：

```bash
python scripts/build_stage2_mixture.py \
  --aishell data/aishell1/processed \
  --sources data/external_speech_instructions \
  --output data/stage2_mixture \
  --total 5000
```

可选外部文件放在 `data/external_speech_instructions/`：`meeting.jsonl`、`instruction.jsonl`、`understanding.jsonl`，格式与 `speech_instructions/*.jsonl` 相同。脚本按 50% ASR、20% 会议、20% 指令、10% 理解分配配额，并输出 `train.jsonl`、`dev.jsonl` 和 `metadata.json`。外部文件缺失时会明确标记 `aishell1_fallback`，不会把 AISHELL-1 派生样本伪装成真实问答数据。

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

当前训练脚本记录 CTC loss。完整评估使用 CTC 贪心解码并计算 CER：

~~~text
checkpoint → log-Mel → Conformer + CTC
          → greedy CTC decode → 中文文本 → CER
~~~

推荐使用一键报告脚本。它会一次性评估 dev/test、比较多个 checkpoint、统计参数量和模型前向推理速度，并保存文本样例和错误案例：

~~~bash
# 先用 1 个 checkpoint 做快速检查
python scripts/evaluate_conformer_report.py \
  --data data/aishell1/processed \
  --output outputs/02_acoustic_encoder \
  --split both \
  --max-checkpoints 1 \
  --samples 20 \
  --batch-size 16

# 完整比较 outputs/02_acoustic_encoder 下的所有 checkpoint
python scripts/evaluate_conformer_report.py \
  --data data/aishell1/processed \
  --output outputs/02_acoustic_encoder \
  --split both \
  --batch-size 16
~~~

输出文件位于 `outputs/02_acoustic_encoder/`：

- `checkpoint_comparison.csv`：每个 checkpoint 在 dev/test 上的 CER、编辑距离、参数量、模型前向耗时、RTF 等；
- `evaluation_samples.csv`：参考文本与预测文本样例；
- `evaluation_errors.csv`：按单条样本 CER 排序的错误案例，便于定位替换、删除和插入；
- `evaluation_report.json`：以上结果及每个 split 的最佳 checkpoint 汇总。

其中 `inference_seconds` 只计模型前向计算，不含音频读取和 log-Mel 特征提取；`audio_duration_seconds_estimate` 根据 10 ms 的特征帧移估算，`real_time_factor < 1` 表示模型前向速度快于音频播放速度。

评估已有 checkpoint：

~~~bash
python scripts/evaluate_conformer_ctc.py \
  --data data/aishell1/processed \
  --checkpoint outputs/02_acoustic_encoder/checkpoint_epoch_020.pt \
  --split dev
~~~

结果会保存为 `dev_metrics.json` 和 `dev_predictions.csv`；将 `--split` 改为 `test` 即可评估测试集。`cer` 越低越好。

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
