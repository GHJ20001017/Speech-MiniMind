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

训练样例可以写成：

```text
语音 embedding + “请将这段语音转写为文字：” + “今天天气很好”
```

输入 embedding 的排列为：

```text
[speech_1, ..., speech_M, prompt_1, ..., prompt_N, target_1, ..., target_K]
```

损失只计算目标文本部分。`-100` 是 PyTorch 交叉熵的 ignore index：

```text
labels = [-100, ..., -100,  # 语音前缀和 prompt
          target_1, ..., target_K]
```

## 4. 运行代码

MiniMind 的源码和权重由官方仓库维护，本项目不复制其大模型文件。请先下载一个 Transformers 格式的 MiniMind 模型目录，再运行：

```bash
python scripts/train_speech_projector.py \
  --data data/aishell1/processed \
  --encoder-checkpoint outputs/02_acoustic_encoder/tiny_conformer_ctc.pt \
  --minimind-model /path/to/minimind-3 \
  --output outputs/03_speech_minimind \
  --epochs 3 \
  --batch-size 2
```

第一次建议使用 `--limit 16` 做 smoke test。默认冻结 Tiny Conformer 和 MiniMind，只更新 Projector，便于先观察数据形状、loss 和显存占用。

## 5. 这一步还不是什么

AISHELL-1 只有语音和转写文本，没有“听完语音后回答问题”的标注。因此本章得到的是语音条件的转写桥接模型，还不是完整的 Speech LLM。下一步需要加入语音指令数据，训练问答、分类和信息抽取，并保留 CTC loss 作为辅助目标。

可以先运行 `scripts/prepare_speech_instructions.py` 生成第二阶段统一格式：

```bash
python scripts/prepare_speech_instructions.py \
  --input data/aishell1/processed \
  --output data/speech_instructions \
  --include-text-ops
```

输出为 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `metadata.json`。主任务是 `transcription`；可选的 `char_count`、`first_character`、`last_character` 是由转写文本直接验证的辅助指令任务。它们用于先验证数据接口和 instruction-following，不能替代真实的语音摘要、问答或意图识别标注。

为了让小模型先跑通第二阶段，可以使用 `scripts/build_stage2_mixture.py` 生成 5,000 条左右的混合数据。它预留 50% 中文 ASR、20% 会议、20% 语音指令、10% 理解任务的配额，并从 `data/external_speech_instructions/` 读取后面补充的真实 JSONL。外部数据没有准备好时，脚本会使用带有 `aishell1_fallback` 标记的转写样本填充空缺，方便测试数据管线，但这些样本不能用于宣称模型已经学会语音问答。
