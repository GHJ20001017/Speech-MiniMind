# 01. Mel 频谱：把 STFT 变成语音特征

上一章中，一帧 25 ms 的音频经过 FFT 得到 301 个线性频率 bin。整段音频经过滑动窗口后，得到 STFT：

```text
STFT.shape = (时间帧数, 频率 bin 数) = (1274, 301)
```

这一章继续把它转换成语音模型常用的 `log-Mel spectrogram`。

## 1. 为什么还要 Mel 化

线性频率轴的每个 bin 间隔相同，例如 0、40、80、120 Hz。人耳对低频差异更敏感，对高频差异相对不敏感；语音的很多重要结构也集中在较低频率区域。Mel 尺度让低频区域分配更多滤波器、高频区域分配更宽的滤波器。

常用转换公式是：

```text
mel = 2595 × log10(1 + hz / 700)
hz  = 700 × (10^(mel / 2595) - 1)
```

Mel 不是新的声音频率，而是频率轴的一种重新标尺。

## 2. 三角 Mel 滤波器

我们在 0 Hz 到 Nyquist 频率之间均匀取 `n_mels + 2` 个 Mel 点。相邻三个点组成一个三角滤波器：左边从 0 线性上升到 1，右边从 1 线性下降到 0。

```text
        /\
       /  \
------/----\------  频率
    left center right
```

每个滤波器会对附近的 STFT 功率做加权求和。80 个三角滤波器把 301 个线性频率 bin 压缩成 80 个 Mel 能量：

```text
power_spectrum (301)
        × Mel filter bank (80 × 301)
        ↓
mel_energy (80)
```

这不是简单地删除频率，而是用重叠的三角权重聚合邻近频率。

## 3. 为什么使用功率谱

FFT 的复数系数包含幅度和相位。Mel 特征主要关心每个频带有多少能量，因此先取幅度平方：

```text
power[k] = |X[k]|²
```

再用滤波器组聚合：

```text
mel_energy[m] = Σk power[k] × filter[m, k]
```

## 4. 为什么取对数

不同频带之间的能量可能相差很多。直接使用线性能量时，强能量会掩盖弱但有用的结构，因此取对数：

```text
log_mel[m] = log(max(mel_energy[m], ε))
```

`ε` 防止对 0 取对数。这里的 log 是自然对数；如果使用 dB，也可以写成 `10 × log10(power)`。

## 5. 运行可视化

```bash
python scripts/analyze_audio.py examples/disgusted_to_happy.wav \
  --mel-plot outputs/mel.png
```

图中展示三个中间产物：

1. 80 个 Mel 三角滤波器，以及它们覆盖的线性频率范围。
2. 一个时间帧的功率频谱和经过滤波器组后的 Mel 能量。
3. 全部时间帧组成的 log-Mel spectrogram。

终端会打印：

```text
mel_shape: (1274, 80)
```

它表示 1274 个时间帧，每帧 80 个 Mel 频带。这个二维数组就是后续 Conformer 或 Transformer 声学编码器常用的输入形式。

## 6. 与 STFT 的对比

```text
波形 → 分帧 → Hann 窗 → FFT → 功率谱
                              ↓
                    Mel 滤波器组 → log
                              ↓
                       log-Mel 特征
```

STFT 保留了较细的线性频率信息；log-Mel 用更符合听觉和语音任务的频带重新组织这些信息，并减少特征维度。下一章会把 `(时间帧数, Mel 频带数)` 输入一个小型声学编码器。
