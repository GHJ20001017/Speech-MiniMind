# 03. Speech Projector：把声学编码器接入 Qwen3-0.6B

## 1. 本章目标

02 结束后，我们已经有了一个能把中文语音转换成 Conformer 隐藏状态的声学编码器：

```text
log-Mel:        [batch, T, 80]
Conformer:      [batch, T/4, 256]
```

Qwen3-0.6B 接收的却是文本 token embedding，其 hidden size 为 1024：

```text
文本 embedding: [batch, N, 1024]
```

因此需要一个中间模块，把两种表示对齐。这个模块叫 **Speech Projector**。

## 2. 为什么不能直接拼接

`[T/4, 256]` 和 `[N, 1024]` 的最后一维不同，不能直接 `torch.cat`。Projector 用可训练参数学习声学空间和语言空间之间的映射：

```text
[T/4, 256]
    ↓ 一维卷积，时间降采样
[T/16, 256]
    ↓ LayerNorm + Linear + GELU + Linear
[T/16, 1024]
```

降采样很重要。AISHELL-1 中一段十几秒的语音可能有上千个声学帧，如果每一帧都变成一个语言模型 token，注意力计算和位置长度都会快速增加。

> 换成 Qwen3-0.6B 后输出维度是 **1024**，而 MiniMind-3 是 768。两者的 projector 权重形状不同，**旧的 `outputs/03_speech_minimind_projector/projector_epoch_*.pt` 无法复用**，必须按本章重训。（`llm_dim` 由 `lm.config.hidden_size` 自动决定，训练脚本也会在加载时校验 projector 与 backbone 的维度是否一致。）

## 3. 语音前缀如何进入语言模型

训练样例按 Qwen3 的 chat template 排布，语音占据 `user` 轮的内容位。本章与第 04 章都使用**非思考模式**，在 Qwen3 里这意味着模板在 `assistant` 后面预先放一个**已经闭合的空 think 块**，模型因此直接作答，不会展开推理：

```text
<|im_start|>system
请转写为中文<|im_end|>
<|im_start|>user
<|audio_start|>{语音}<|audio_end|><|im_end|>
<|im_start|>assistant
<think>

</think>

今天天气很好<|im_end|>
```

`{语音}` 是 Projector 输出的连续 speech embedding，不是 token id，所以文本被拆成语音前后两段。`<|audio_start|>` / `<|audio_end|>` 在 Qwen3 的 tokenizer 里**并不存在**：直接把它们交给 tokenizer 会被拆成 6 个互不相关的字节 token（`<|audio_start|>` → `[27, 91, 16736, 4906, 91, 29]`）。因此 `model/chat_format.py` 把它们追加为 added special token，在 Qwen3-0.6B 上落在 **151669 / 151670 / 151671**——仍在 embedding 表的 151936 行之内，所以**不需要扩表**，只是把这些原本闲置的行改成两个 marker 行（用全体文本 embedding 的均值初始化）。路线 B（`model/audio_lm.py`）复用同一对，两条路线对「音频在此」的信号因此一致。

```text
[system 轮 + user 轮开头 + <|audio_start|>, speech_1, ..., speech_M, <|audio_end|> + user 轮结尾 + assistant 轮开头 + 空 think 块, target_1, ..., target_K]
```

损失只计算目标文本部分。`-100` 是 PyTorch 交叉熵的 ignore index：

```text
labels = [-100, ..., -100,  # system/user 前缀、<|audio_start|>、语音前缀、<|audio_end|>、assistant 轮开头、空 think 块
          target_1, ..., target_K]   # {answer}<|im_end|>
```

这与 Qwen3 官方的 SFT 约定一致：loss 从 `<|im_start|>assistant\n` 之后开始（在本项目的布局里，就是那个空 think 块之后），并且把结束标记 `<|im_end|>` 计入 loss。`model/chat_format.py` 在每次加载模型时都会断言这套手工拼出的布局与 `tokenizer.apply_chat_template(..., enable_thinking=False)` 的 **token id 完全一致**，避免 tokenizer 更新后训练与推理悄悄错位。

## 4. 运行代码

先在本地准备一份 Qwen3-0.6B 权重（本项目的 95 机器上放在 `/gpu3/guhj/models/Qwen3-0.6B`），Qwen3 是 transformers 原生结构，不需要 `trust_remote_code`。

`--data` 指向第 2 节下载的 stage-1 目录（内含 AISHELL-1 官方划分的 `train.jsonl` / `dev.jsonl` / `test.jsonl`，训练时不再切分）：

```bash
python trainer/train_speech_projector.py \
  --data data/speech2text_corpus/stage1_aishell \
  --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \
  --qwen3-model /gpu3/guhj/models/Qwen3-0.6B \
  --output outputs/03_speech_qwen3_projector \
  --epochs 3 \
  --batch-size 2
```

第一次建议使用 `--limit 16` 做 smoke test。默认冻结 Tiny Conformer 和 Qwen3，只更新 Projector，便于先观察数据形状、loss 和显存占用。

## 5. 这一步还不是什么

AISHELL-1 只有语音和转写文本，没有“听完语音后回答问题”的标注。因此本章得到的是语音条件的转写桥接模型，还不是完整的 Speech LLM。下一步需要加入语音问答数据，训练问答、分类和信息抽取。

第二阶段数据（stage 2）来自 ModelScope 仓库，已把 AISHELL-1 之外的自然问答 / 指令来源（COIG、Firefly、moss_speech_qa、VoiceAssistant-400K 等）筛选、去重、统一成同一行格式并切分好，见 README 第 2 节。

有了第二阶段的语音问答数据，就可以进入下一章微调整个 Qwen3-0.6B（LoRA 或全参），并同步以小学习率训练 Projector。
