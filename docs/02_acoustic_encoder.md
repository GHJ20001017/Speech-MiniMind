# 02. 声学编码器：从 log-Mel 到中文语音识别

前两章完成了信号处理部分：

```text
WAV 波形 → 分帧加窗 → STFT → Mel 滤波器组 → log-Mel
```

本章把每一帧的 80 维 log-Mel 特征送入一个 Tiny Conformer，使用 AISHELL-1 中文语音数据集训练字符级 CTC 语音识别模型。目标不是立刻得到商用 ASR，而是把“声学特征 → 编码器 → 中文文字”的完整链路跑通。

## 1. 准备环境

本地开发环境：

```bash
conda activate speech-llm
cd Speech-MiniMind
python -m pip install -r requirements.txt
```

95 服务器环境：

```bash
source /gpu/anaconda3/etc/profile.d/conda.sh
conda activate /gpu3/guhj/envs/speech-llm
cd /gpu3/guhj/Speech-MiniMind
```

服务器环境包含 Python 3.12、PyTorch 2.10.0+cu128，并可以使用 A800 GPU。后文的 `python` 均指当前已经激活的环境。

## 2. 下载 AISHELL-1

AISHELL-1 是中文普通话朗读语料，包含训练、开发和测试划分。下载脚本从 OpenSLR 获取数据，并解压到 `data/aishell1`：

```bash
python scripts/download_aishell1.py
```

脚本使用 `curl` 的断点续传和自动重试；如果网络中断，直接再次运行同一命令即可从已有文件继续。下载完成后会创建 `.extracted` 标记，再次运行时不会重复下载或解压。目录大致如下：

```text
data/aishell1/
├── data_aishell/
│   ├── wav/
│   ├── transcript/
│   └── resource_aishell/
└── data_aishell.tgz
```

## 3. 生成 manifest 和词表

训练脚本需要 CSV manifest 和字符词表：

```bash
python scripts/prepare_aishell1.py
```

结果位于 `data/aishell1/processed`：

```text
processed/
├── train.csv
├── dev.csv
├── test.csv
└── vocab.txt
```

每一行 CSV 记录音频路径和中文文本。`vocab.txt` 是字符级词表，第 0 行固定为 `<blank>`，这是 CTC 使用的空白符；后面的每一行是一个汉字。模型直接学习“声学帧 → 汉字序列”，不需要先做分词。

## 4. 一条样本如何变成模型输入

训练时沿用前面章节的参数：

```text
采样率       16 kHz（AISHELL-1 原始音频）
窗口长度     25 ms = 400 个采样点
帧移         10 ms = 160 个采样点
Mel 频带数   80
```

一段音频会得到二维矩阵：

```text
log-Mel.shape = (帧数, 80)
batch.shape   = (batch_size, 最大帧数, 80)
```

不同音频时长不同，组成 batch 时会在末尾补零，并保留每个样本的真实帧数。padding mask 会告诉注意力层哪些位置是补出来的，避免补零帧参与计算。

## 5. Tiny Conformer 结构

当前配置约 9M 参数，便于单张 GPU 训练，也保留后续复用价值：

```text
(batch, time, 80) log-Mel
        ↓ 两层 Conv2d(kernel=3, stride=2)
(batch, time/4, 19, 256)
        ↓ 展平频率维 + Linear
(batch, time/4, 256)
        ↓ 4 个 Conformer block
(batch, time/4, 256)
        ↓ Linear(vocab_size)
(batch, time/4, vocab_size) logits
```

每个 block 包含半步前馈网络、Multi-Head Self-Attention、深度可分离卷积、半步前馈网络和 LayerNorm。

| 参数 | 当前值 |
|---|---:|
| Conformer 层数 | 4 |
| hidden dimension | 256 |
| 注意力头数 | 4 |
| FFN dimension | 1024 |
| 卷积 kernel size | 31 |
| 时间下采样 | 4 倍 |
| 总参数量 | 约 9.2M（随词表大小变化） |

下采样不是简单地把长度除以 4。两次无 padding 卷积的真实输出长度是：

```text
L1 = floor((L - 3) / 2) + 1
L2 = floor((L1 - 3) / 2) + 1
```

训练脚本使用这个长度传给 CTC，保证变长音频的损失计算正确。

## 6. CTC 在解决什么问题

音频帧数通常比汉字数多，但数据集没有标出每个汉字对应哪些帧。CTC 允许模型输出更长的帧级序列，其中可以出现 `<blank>` 和重复字符；解码时合并连续重复并删除 blank：

```text
帧级输出：我  <blank>  我  爱  <blank>  北  京
CTC 解码：我爱北京
```

训练时，`CTCLoss` 比较整段帧级 logits 和目标汉字序列，自动汇总所有可能的对齐路径。因此只需要音频级文本，不需要人工标注每个字的起止时间。

## 7. 开始训练

确认 `train.csv`、`vocab.txt` 已生成后运行：

```bash
python scripts/train_conformer_ctc.py \
  --data data/aishell1/processed \
  --epochs 20 \
  --batch-size 8 \
  --lr 2e-4
```

脚本会自动选择 CUDA，打印每轮 CTC loss，并保存：

```text
data/aishell1/processed/tiny_conformer_ctc.pt
```

checkpoint 包含 `model`（编码器和 CTC head 参数）及 `vocab`（字符到整数 id 的映射）。后续做语音理解时可以只加载 `model.encoder`，再替换任务头或接入 MiniMind。

## 8. 如何判断训练是否有效

训练 loss 下降只能说明模型拟合了训练集，不能代表识别准确率。下一步应补充贪心 CTC 解码，并在 `dev.csv`、`test.csv` 上计算 CER（字符错误率）：

```text
CER = (替换数 + 删除数 + 插入数) / 参考文本字符数
```

本章先完成可复现的训练闭环；评估脚本和 beam search 解码将在后续章节加入。

## 9. 常见问题

**显存不足怎么办？** 将 `--batch-size` 从 8 降到 4 或 2。长样本会决定一个 batch 的显存占用。

**为什么没有 dev/test 训练循环？** 当前脚本是最小教学版本，只训练 `train.csv` 并保存 checkpoint；开发集和测试集留给后续 CER 评估。

**为什么参数量不是固定的 8M？** CTC head 的参数量是 `hidden_dim × vocab_size`。词表不同，最终总参数量会略有变化；当前 AISHELL-1 配置约为 9.2M。
