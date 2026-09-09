# Speech-MiniMind

从一段 WAV 开始，亲手构建一个中文 Speech LLM：WAV → FFT → Mel → Tiny Conformer + CTC → Speech Projector → MiniMind。

目标：能看懂、能运行、能修改的整套教学流水线。

```text
00 语音基础 → 01 Mel 频谱 → 02 声学编码器（Tiny Conformer + CTC，含流式版） → 03 接入 MiniMind
```

分章教学文档见 [`docs/`](docs/)：

| 章节 | 内容 | 文档 |
|---|---|---|
| 00 语音基础 | WAV、波形、FFT、STFT | [docs/00_audio_basics.md](docs/00_audio_basics.md) |
| 01 Mel 频谱 | 功率谱、Mel 滤波器组、log-Mel | [docs/01_mel_spectrogram.md](docs/01_mel_spectrogram.md) |
| 02 声学编码器 | Tiny Conformer、AISHELL-1、CTC；流式 Conformer（因果分块版） | [docs/02_acoustic_encoder.md](docs/02_acoustic_encoder.md) |
| 03 接入 MiniMind | Speech Projector、语音前缀 | [docs/03_speech_minimind.md](docs/03_speech_minimind.md) |

## 环境安装

```bash
conda create -n speech-llm python=3.11
conda activate speech-llm
python -m pip install -r requirements.txt
```

### 1. 语音分析（00/01）

对示例音频生成波形、频谱、STFT 动画：

```bash
python scripts/analyze_audio.py examples/disgusted_to_happy.wav \
  --plot outputs/example.png --stft-plot outputs/stft.png --stft-gif outputs/stft_process.gif
```

### 2. AISHELL-1 数据准备（02）

```bash
# 下载数据（国内用 ModelScope 镜像，支持断点续传）
python scripts/download_aishell1.py

# 生成 train/dev/test.csv 和 vocab.txt
python scripts/prepare_aishell1.py
```

### 3. 训练中文声学编码器（02，Tiny Conformer + CTC）

#### 非流式（离线整句识别）

```bash
python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed --epochs 20 --batch-size 32 --lr 2e-4
```

输出到 `outputs/02_acoustic_encoder/`：`metrics.csv`、`loss_curve.png`、逐 epoch checkpoint、`tiny_conformer_ctc.pt`。

#### 流式（chunk-based 因果版）

同一编码器任务的流式实现，用于边听边出的实时场景：

+ **模型**：`model/conformer_streaming.py`（因果下采样、因果卷积、分块因果注意力 + 左上下文缓存）、`model/ctc_streaming.py`（流式 CTC 封装）
+ **训练**：

```bash
python scripts/train_conformer_streaming_ctc.py \
  --data data/aishell1/processed --epochs 20 --batch-size 32 --lr 2e-4 \
  --chunk-size 32 --left-context 16
```


### 4. 评估声学编码器（02）

```bash
# 一键报告：dev/test CER、checkpoint 对比、样例、RTF
python scripts/evaluate_conformer_report.py \
  --data data/aishell1/processed --output outputs/02_acoustic_encoder --split both

# 单 checkpoint 评估
python scripts/evaluate_conformer_ctc.py \
  --data data/aishell1/processed \
  --checkpoint outputs/02_acoustic_encoder/checkpoint_epoch_020.pt --split dev
```

同目录还有 `scripts/plot_training_metrics.py` 可绘制训练曲线。

### 5. 构建第二阶段的语音指令数据（03）

把 AISHELL-1 转写标注转成统一的语音指令格式：

```bash
python scripts/prepare_speech_instructions.py \
  --input data/aishell1/processed --output data/speech_instructions
```

每行 JSON：`{"audio": "...", "instruction": "请将这段语音准确转写为中文文本。", "answer": "...", "task": "transcription"}`。

再混合外部指令数据构建小规模训练集：

```bash
python scripts/build_stage2_mixture.py \
  --aishell data/aishell1/processed --sources data/external_speech_instructions \
  --output data/stage2_mixture --total 5000
```

（`data/external_speech_instructions/` 下可选放 `meeting.jsonl`、`instruction.jsonl`、`understanding.jsonl`。）

### 6. 语音接入 MiniMind（03，训练 Projector）

先下载 [MiniMind Transformers 权重](https://github.com/jingyaogong/minimind)（如 `minimind-3`）到本地目录，然后：

```bash
python scripts/train_speech_minimind.py \
  --data data/aishell1/processed \
  --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \
  --minimind-model /path/to/minimind-3 \
  --output outputs/03_speech_minimind --epochs 3 --batch-size 2
```

冻结 Conformer 和 MiniMind，只训练约 0.8M 参数的 `SpeechProjector`。

### 7. 构建并合成语音问答数据（用于下一阶段）

```bash
# 从 moss-003 SFT 抽取中文多轮子集
python scripts/prepare_moss_speech_qa.py --input /path/to/moss.zip --output data/moss_speech_qa

# 用 Qwen3-TTS 把 instruction 文本合成为真实中文音频
python scripts/generate_moss_speech_qa_tts.py --data data/moss_speech_qa

# 从 VoiceAssistant-400K 随机抽样并下载本地音频
python scripts/prepare_voiceassistant_400k.py --num-samples 50000 --output data/voiceassistant400k_50k
```

## 模型配置（02 Tiny Conformer）

| 项 | 值 |
|---|---:|
| 输入 | 80 维 log-Mel |
| 层数 / hidden / heads | 4 / 256 / 4 |
| FFN | 1024 |
| 卷积 kernel | 31 |
| 时间下采样 | 4× |
| 参数量 | 约 9.0M（随词表变） |
| 训练目标 | 字符级 CTC |

## 在 GPU 服务器训练

```bash
source /gpu/anaconda3/etc/profile.d/conda.sh
conda activate /gpu3/guhj/envs/speech-llm
cd /gpu3/guhj/Speech-MiniMind

CUDA_VISIBLE_DEVICES=7 python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed --epochs 20 --batch-size 32 --lr 2e-4
```

后台运行并看日志：

```bash
mkdir -p outputs/02_acoustic_encoder
nohup env CUDA_VISIBLE_DEVICES=7 python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed --epochs 20 --batch-size 32 --lr 2e-4 \
  > outputs/02_acoustic_encoder/train.log 2>&1 &
tail -f outputs/02_acoustic_encoder/train.log
```

## 目录结构

```text
Speech-MiniMind/
├── docs/        # 分章教学文档
├── examples/    # 示例音频
├── model/       # Conformer、CTC、流式版、Projector、MiniMind 适配
├── scripts/     # 分析 / 准备 / 训练 / 评估 / 合成脚本
├── data/        # 本地数据，不提交
├── outputs/     # 图表、日志、checkpoint，不提交
├── requirements.txt
└── README.md
```

## 开源说明

数据集遵循 AISHELL-1 原始许可；`data/`、`outputs/`、`.pt`、压缩包不提交仓库。正式发布前会补充代码许可证与数据集引用。