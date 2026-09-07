# 02. 声学编码器：从 log-Mel 到音频分类

前两章把音频变成了二维特征：

```text
音频 → STFT → Mel 滤波器组 → log-Mel
```

本章第一次把特征送进神经网络，完成一个最小的音频分类任务：判断 1 秒片段是 `speech` 还是 `non_speech`。

## 1. 数据集

为了让教程可以离线复现，数据集脚本使用仓库中的 `examples/disgusted_to_happy.wav`：

- `speech=1`：从真实语音中随机裁剪 1 秒片段；
- `non_speech=0`：生成不同频率的低幅纯音并叠加少量噪声。

这不是用于发表结果的 benchmark，而是一个能让初学者看懂训练链路的 toy dataset。它的目的，是让模型先学会区分明显不同的声学模式。

生成数据：

```bash
python scripts/make_audio_classification_dataset.py \
  --source examples/disgusted_to_happy.wav \
  --output data/audio_classification \
  --clips-per-class 40
```

输出目录包括：

```text
data/audio_classification/
├── speech_0000.wav ...
├── non_speech_0000.wav ...
├── train.csv
└── valid.csv
```

manifest 中每一行记录音频路径、整数标签和类别名。固定随机种子后，数据可以重复生成。

## 2. 单个样本的特征形状

每个 1 秒、24 kHz 的音频片段经过和前面相同的参数：

```text
25 ms 窗口，10 ms 帧移，80 个 Mel 频带
```

得到约 99 个时间帧：

```text
一个样本：log-Mel.shape = (99, 80)
一个 batch：features.shape = (batch, 99, 80)
```

第一维是时间，第二维是 Mel 频带。模型不会把整段 1 秒音频压成一个数字，而是先逐帧处理，再学习时间上下文。

## 3. 模型结构

本章使用一个容易追踪的声学编码器：

```text
(batch, time, 80)
        ↓ Linear(80 → 128)
(batch, time, 128)
        ↓ ReLU + Linear(128 → 128)
(batch, time, 128)
        ↓ Conv1d(kernel=5)
(batch, time, 128)
        ↓ 沿时间取平均 mean(dim=1)
(batch, 128)
        ↓ Linear(128 → 2)
(batch, 2) logits
```

`Conv1d(kernel=5)` 让每一帧可以利用附近约 5 帧的局部上下文；平均池化把可变长度的时间序列变成一个固定长度向量；最后的线性层输出两个类别的 logits。

## 4. 训练

先安装 PyTorch：

```bash
python -m pip install -r requirements.txt
```

然后运行：

```bash
python scripts/train_audio_classifier.py \
  --data data/audio_classification \
  --epochs 8
```

每轮会输出训练 loss 和验证集准确率。交叉熵损失会比较模型输出的两个 logits 与真实标签，反向传播则更新 Linear 和 Conv1d 的参数。

## 5. 从一个样本看完整链路

```text
1 秒 WAV
  → 读取、归一化
  → 分帧、加窗、FFT
  → 功率谱、Mel 聚合、log
  → (99, 80) log-Mel
  → Linear/Conv1d
  → (99, 128) 隐藏表示
  → 时间平均池化
  → (128,) 音频向量
  → 分类器
  → [speech logit, non_speech logit]
```

本章结束后，已经完成了“手工音频特征 → 神经网络 → 任务输出”的闭环。下一章可以把分类头替换成 CTC，并将每个时间帧映射到文字 token，开始真正的 ASR。

## 6. 中文 Tiny Conformer 训练

如果目标是训练后续可复用的中文编码器，请使用 AISHELL-1 的 CTC 任务：

```bash
python scripts/download_aishell1.py
python scripts/prepare_aishell1.py
python scripts/train_conformer_ctc.py --data data/aishell1/processed
```

当前配置是 4 层、hidden=256、4 个注意力头、卷积 kernel=31。训练脚本会打印实际参数量，CTC 输出层的大小会随 AISHELL-1 字符表变化。训练完成后，`tiny_conformer_ctc.pt` 中的 `model.encoder` 就是可以在后续任务中复用的中文声学编码器。正式评估时应在 AISHELL-1 dev/test 上计算 CER，而不是只看训练 loss。
