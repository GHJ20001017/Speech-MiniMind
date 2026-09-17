# 03. Speech Projector：把声学编码器接入 MiniMind

## 1. 本章目标

02 结束后，我们已经有了一个能把中文语音转换成 Conformer 隐藏状态的声学编码器：

```text
log-Mel:        [batch, T, 80]
Conformer:      [batch, T/4, 256]
```

MiniMind 接收的却是文本 token embedding，例如 hidden size 为 768：

```text
文本 embedding: [batch, N, 768]
```

因此需要一个中间模块，把两种表示对齐。这个模块叫 **Speech Projector**。

## 2. 为什么不能直接拼接

`[T/4, 256]` 和 `[N, 768]` 的最后一维不同，不能直接 `torch.cat`。Projector 用可训练参数学习声学空间和语言空间之间的映射：

```text
[T/4, 256]
    ↓ 一维卷积，时间降采样
[T/16, 256]
    ↓ LayerNorm + Linear + GELU + Linear
[T/16, 768]
```

降采样很重要。AISHELL-1 中一段十几秒的语音可能有上千个声学帧，如果每一帧都变成一个语言模型 token，注意力计算和位置长度都会快速增加。

## 3. 语音前缀如何进入语言模型

训练样例按 MiniMind 原生 chat template 排布，语音占据 `user` 轮的内容位：

```text
<|im_start|>system
请转写为中文<|im_end|>
<|im_start|>user
{语音}<|im_end|>
<|im_start|>assistant
今天天气很好<|im_end|>
```

`{语音}` 是 Projector 输出的连续 speech embedding，不是 token id，所以文本被拆成语音前后两段：

```text
[system 轮 + user 轮开头, speech_1, ..., speech_M, user 轮结尾 + assistant 轮开头, target_1, ..., target_K]
```

损失只计算目标文本部分。`-100` 是 PyTorch 交叉熵的 ignore index：

```text
labels = [-100, ..., -100,  # system/user 前缀、语音前缀、assistant 轮开头
          target_1, ..., target_K]   # {answer}<|im_end|>
```

这与 MiniMind 官方 `SFTDataset.generate_labels` 的约定一致：loss 从 `<|im_start|>assistant\n` 之后开始，并且把结束标记 `<|im_end|>` 计入 loss。

## 4. 运行代码

MiniMind 的源码和权重由官方仓库维护，本项目不复制其大模型文件。请先下载一个 Transformers 格式的 MiniMind 模型目录，再运行：

`--data` 指向第 2 节下载的 stage-1 目录（内含 AISHELL-1 官方划分的 `train.jsonl` / `dev.jsonl` / `test.jsonl`，训练时不再切分）：

```bash
python trainer/train_speech_projector.py \
  --data data/speech2text_corpus/stage1_aishell \
  --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \
  --minimind-model /path/to/minimind-3 \
  --output outputs/03_speech_minimind \
  --epochs 3 \
  --batch-size 2
```

第一次建议使用 `--limit 16` 做 smoke test。默认冻结 Tiny Conformer 和 MiniMind，只更新 Projector，便于先观察数据形状、loss 和显存占用。

## 5. 这一步还不是什么

AISHELL-1 只有语音和转写文本，没有“听完语音后回答问题”的标注。因此本章得到的是语音条件的转写桥接模型，还不是完整的 Speech LLM。下一步需要加入语音问答数据，训练问答、分类和信息抽取。

第二阶段数据（stage 2）来自 ModelScope 仓库，已把 AISHELL-1 之外的自然问答 / 指令来源（COIG、Firefly、moss_speech_qa、VoiceAssistant-400K 等）筛选、去重、统一成同一行格式并切分好，见 README 第 2 节。

有了第二阶段的语音问答数据，就可以进入下一章微调整个 MiniMind（LoRA 或全参），并同步以小学习率训练 Projector。
