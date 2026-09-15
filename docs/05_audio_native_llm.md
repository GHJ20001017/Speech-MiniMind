# 05. 音频专属 LLM：离散 codebook 端到端（路线 B）

## 1. 与路线 A 的区别

路线 A 是**级联式**：冻结声学编码器给出**连续**帧级向量 → Projector → MiniMind → **文本**。

路线 B 是**原生音频**：语音被量化成**离散 codebook token**，同一个 LLM 直接在 token 序列上建模并生成，再由解码器还原波形：

```text
WAV ──► 冻结 codec 编码器 ──► 离散 audio tokens ──► 音频专属 LLM
                                                    │
                              WAV ◄── 冻结 codec 解码器 ◄── 生成的 audio tokens
```

关键差别只有一句：**输入输出都是同一套 codec 的离散 token**，因此

- 必须**扩词表**，把 `codebook_size × num_codebooks` 个音频 token 追加到文本词表之后；
- 必须训练 `embed_tokens` / `lm_head` 的**新增行**，LoRA 覆盖不到它们（所以路线 B 默认 `--tune full`）；
- 句子变短了：Mimi 一帧 80 ms、一帧 8 个 code，30 秒语音 ≈ 375 帧，比路线 A 的连续前缀更紧凑。

## 2. codec：先量化重建上限（M0 门槛）

路线 B 不对 codec 做任何训练。本项目支持两个后端（`model/audio_codec.py`）：

| 后端 | codebook | 帧率 | 采样率 | 说明 |
|---|---|---|---|---|
| `mimi`（默认） | 8 × 2048 | 12.5 Hz | 24 kHz | Kyutai Mimi，与 MiniMind-O 的 `sft_a2a` 数据**同源**，可直接复用其已 token 化的音频 |
| `encodec` | 8 × 1024 | 75 Hz | 24 kHz | Meta EnCodec 24kHz，作为对照 |

> **注意**：公开的 Mimi 权重（`kyutai/mimi`、ModelScope `gongjy/mimi`）实际带 **32** 个 quantizer（1 语义 + 31 声学），而 MiniMind-O 和 `sft_a2a` 只用**前 8** 个。`MimiCodec` 默认 `num_codebooks=8`，并把它传给 `encode(num_quantizers=...)`，让输入语音和数据集里的回答 code 来自同一组子码本。若改成其他数量，两侧必须一致。

进入训练前必须先量化「量化损失」，否则模型再强也救不回差 codec：

```bash
python scripts/eval_codec_reconstruction.py \
  --data data/aishell1/processed --split dev --num 20 \
  --codec-type mimi --device cuda:0 \
  --output outputs/05_route_b_codec_check
```

输出 `metrics.csv` / `summary.json` 与 `audio/` 下的 `orig_*.wav` / `recon_*.wav` 对照：

- `mel_mae`：log-Mel 重建失真（始终可用）；
- `stoi` / `pesq`：装了 `pystoi` / `pesq` 才有；
- `cer`：把重建音频再送一遍 SenseVoice 做 ASR，与原始转写算字符错误率——**这是"还能不能听懂"的最终判据**。

**门槛**：中文可懂、无明显金属音。不达标就换 `--codec-type encodec`，或换更大的码本。

## 3. 词表布局与训练目标

`model/audio_lm.py` 定义音频 token 的排布。音频块紧跟在文本词表之后：

```text
[0, audio_offset)                     文本 token（保持预训练权重）
[audio_offset + q*C, +C)              codebook q 的 C 个码（q = 0..Q-1）
```

三个特殊 token 复用 MiniMind-3 tokenizer 自带的 `<|audio_start|>`(14) / `<|audio_end|>`(15) / `<|audio_pad|>`(16)（`register_audio_special_tokens` 会发现已存在而跳过；换成普通文本 tokenizer 时才追加为 `additional_special_tokens`）。因此它们位于 `audio_offset` **之下**，`total_vocab_size` 取「音频块末尾」与「特殊 token 最大 id + 1」两者的较大值。

一帧 `t` 的 `Q` 个 code 被**展平**成 `t*Q + q` 的连续 token（与 MiniMind-O 的 `answer_audios` 一致）。训练序列：

```text
[BOS] <|audio_start|> 输入语音 token <|audio_end|>
      <|audio_start|> 输出语音 token <|audio_end|> [EOS]
```

损失**只算输出语音那段**（含它的 `<|audio_end|>`）：`build_audio_batch` 把其余位置全部标成 `-100`。这与路线 A 只算 `answer` 文本完全同构。

## 4. 数据

### 4.1 公开数据：MiniMind-O `sft_a2a`（推荐起点）

MiniMind-O 在 ModelScope 开源了 `gongjy/minimind-o_dataset`，其中 `sft_a2a.parquet` 是**已经 token 化**的语音问答对（`question_audios` 音频字节 + `answer_audios` 展平的 8 层 Mimi codes）。它省掉了全部 TTS 与编码开销，是跑通路线 B 的最短路径：

```bash
python scripts/prepare_speech_to_speech.py --download \
  --file-name sft_a2a.parquet --lang zh \
  --output data/route_b/s2s --device cuda:0 --batch-size 8
```

脚本做的事情：

1. 读 `conversations` / `answer_audios`，把展平 token 还原成 `(Q, T)`，遇到 stop token（`>= codebook_size`）即截断；
2. 用同源 Mimi 编码 `question_audios`，写入 `codes/*_p.npy`；
3. 落盘 `codes/*_a.npy` 与 `data/route_b/s2s/{train,dev}.jsonl`。

**实测事实（95 上跑通的 414024 行 `sft_a2a.parquet`，5.7 GB）**：

- 展平 token 是 **frame-major**：`(t*Q+q)`，即 `arr.reshape(T, 8).T`。按这个顺序解码后送 SenseVoice 往返，转写与原文一致；
  误按 codebook-major（`arr.reshape(8, T)`）解码则完全听不懂。这一点已用 ASR 对照验证，不要把顺序改回去。
- 中文占比约 **34.65%**（`cjk_ratio >= 0.15`），且**分布在文件后段**，所以 `--limit` 做小样本时很容易一条中文都抽不到（表现为 `{"lang": 400}`、`kept: 0`）。要凑中文子集请调大 `--limit` 或跑全量。
- 单个回答的 token 长度是 `Q` 的整数倍（`len % 8 == 0`，stop token 已在数据里被剔除），实测长度 120–23336，token 值域 `[0, 2047]`。
- 读取必须**按 row group 流式**（`ParquetFile.iter_batches`）：`pq.read_table` 会因嵌套列报 `Nested data conversions not implemented for chunked array outputs`，而且会把整份 5.7 GB 读进内存。

先下载 mini 版（`sft_a2a_mini.parquet`）做冒烟测试更省时间。

### 4.2 自有数据：先建 B0 语料，再配对

**B0（纯音频 LM）** 只需要语音，不需要配对。先汇总本仓库已有音频：

```bash
python scripts/prepare_audio_lm_corpus.py --data-root data \
  --output data/route_b/audio_lm            # AISHELL-1(zh) + moss 问题音频
```

再用冻结 codec 一次性编码成离散 token 缓存（**必做**：训练时重跑 codec 会主导步耗）：

```bash
python scripts/cache_audio_tokens.py \
  --data data/route_b/audio_lm --output data/route_b/audio_lm_codes \
  --codec-type mimi --device cuda:0 --batch-size 16
```

`cache_audio_tokens.py` 同时支持 s2s 清单（`prompt_audio` / `answer_audio` 字段），所以自建的「问题语音 + 合成回答语音」配对也走同一条路径。

> 自有 s2s 配对里，回答语音需要用 TTS 合成（可复用 `scripts/generate_moss_speech_qa_tts.py` 的 Qwen3-TTS 约定）。这类数据**音色单一、无噪声、无口语化**，是教学闭环的已知局限，不能当作真实场景泛化结论。

## 5. 训练

### B0：音频 LM 预训练

先让模型学会 codec token 的分布（等价于「音频版文本 LM」）：

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
  trainer/train_audio_lm_pretrain.py \
  --data data/route_b/audio_lm_codes \
  --minimind-model /gpu3/guhj/models/minimind-3 \
  --output outputs/05_route_b_audio_lm --epochs 3 --batch-size 8 \
  --num-codebooks 8 --codebook-size 2048 --max-frames 128 \
  --tune full --num-workers 4 --wandb --wandb-name route_b_b0
```

- 默认 `--tune full`：音频 embedding 行是**新初始化**的，LoRA 碰不到 `embed_tokens`/`lm_head`。
- `--max-frames 128` 把单条压到 128 帧（≈10 s），配合 `--max-length 1152` 控制显存。
- 产出 `metrics.csv`、`loss_curve.png`、逐 epoch 完整模型目录。

### B1/B2：语音到语音指令微调

从 B0 的输出继续，只监督**回答语音**：

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
  trainer/train_speech_to_speech.py \
  --data data/route_b/s2s \
  --init-from outputs/05_route_b_audio_lm/model_epoch_003 \
  --output outputs/06_route_b_s2s --epochs 3 --batch-size 4 \
  --num-codebooks 8 --codebook-size 2048 \
  --max-answer-frames 256 --max-prompt-frames 256 \
  --tune full --num-workers 4 --wandb --wandb-name route_b_s2s
```

- `--init-from` 必须是**已经扩过词表**的 B0 目录；只给 `--minimind-model` 会重新随机初始化音频块（仅用于冒烟）。
- `--code-dropout 0.05` 会随机替换**输入**语音的部分帧，提升对编码噪声的鲁棒性（目标段永不被污染）。
- 每个 epoch 额外记录 `dev_token_acc`：输出语音 token 的 teacher-forcing 准确率。

## 6. 推理

```bash
python scripts/infer_speech_to_speech.py \
  --audio path/to/question.wav \
  --model outputs/06_route_b_s2s/model_epoch_003 \
  --codec-type mimi --device cuda:0 \
  --max-new-frames 300 --output outputs/route_b_answer.wav \
  --print-hypothesis
```

脚本把输入 WAV 编码成 token、自回归生成回答 token、再用同一 codec 解码为 WAV；`--print-hypothesis` 会把生成音频送 ASR 做一次可懂性检查。

**生成必须锁码本**：frame-major 布局下，回答的第 `n` 个 slot 固定属于第 `n % Q` 个码本。`generate_audio_tokens` 会在每一步把 logits 掩蔽到该码本块（外加 `<|audio_end|>`），否则半训好的模型会飘到别的码本块，`from_lm_tokens` 还原出 `>= codebook_size` 的 code 值，Mimi 解码器会直接 `CUDA error: device-side assert triggered`。

## 7. 评测

| 阶段 | 指标 | 脚本 |
|---|---|---|
| M0 codec | `mel_mae`、STOI/PESQ、往返 CER | `scripts/eval_codec_reconstruction.py` |
| M1 B0 | 音频 token CE / perplexity | `metrics.csv` |
| M2 B1 | 回答段 token 准确率、往返 CER | `train_speech_to_speech.py`、`infer_speech_to_speech.py --print-hypothesis` |
| M3 B2 | 多任务指令跟随、端到端可懂性 | 同上 |

## 8. 已知局限

1. **codec 质量是天花板**：请先看 M0 的往返 CER 再解读 B2 的结果，区分「量化损失」与「模型能力」。
2. **合成回答音频**：自有 s2s 配对依赖 TTS，音色与腔调单一。
3. **纯声学 token 语义弱**：若模型收敛但答非所问，可后续加一路文本辅助头（同骨干并行预测回答文本），推理时仍只解码音频。
4. **显存**：请优先单码本或限制帧数；`--max-length` 直接决定序列长度。
