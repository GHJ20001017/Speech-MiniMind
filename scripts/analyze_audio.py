"""Inspect a PCM WAV file and optionally save waveform/spectrum plots."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got {sample_width * 8}-bit")

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def describe(audio: np.ndarray, sample_rate: int) -> None:
    if audio.size == 0:
        raise ValueError("the WAV file contains no samples")
    zero_crossings = np.count_nonzero(np.signbit(audio[1:]) != np.signbit(audio[:-1]))
    duration = audio.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio))))
    zcr = zero_crossings / max(audio.size - 1, 1)
    print(f"sample_rate: {sample_rate} Hz")
    print(f"samples: {audio.size}")
    print(f"duration: {duration:.3f} s")
    print(f"peak_amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"rms: {rms:.4f}")
    print(f"zero_crossing_rate: {zcr:.4f}")


def plot(audio: np.ndarray, sample_rate: int, output: Path) -> None:
    import matplotlib.pyplot as plt

    time = np.arange(audio.size) / sample_rate
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(time, audio, linewidth=0.5)
    axes[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    frequencies = np.fft.rfftfreq(audio.size, 1 / sample_rate)
    axes[1].plot(frequencies, 20 * np.log10(np.maximum(spectrum, 1e-8)), linewidth=0.5)
    axes[1].set(title="Magnitude spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
    axes[1].set_xlim(0, min(sample_rate / 2, 8000))
    figure.savefig(output, dpi=150)
    print(f"plot_saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="path to a 16-bit PCM WAV file")
    parser.add_argument("--plot", type=Path, help="optional output PNG path")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(
            f"audio file not found: {args.audio}\n"
            "Use a real WAV path; 'path/to/example.wav' is only a placeholder."
        )
    audio, sample_rate = read_wav(args.audio)
    describe(audio, sample_rate)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(audio, sample_rate, args.plot)


if __name__ == "__main__":
    main()
