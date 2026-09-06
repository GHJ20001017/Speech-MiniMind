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


def describe(audio: np.ndarray, sample_rate: int, frame_ms: float) -> None:
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
    frame_size = min(audio.size, max(2, round(sample_rate * frame_ms / 1000)))
    print(f"analysis_frame: {frame_size} samples ({frame_size / sample_rate * 1000:.2f} ms)")
    print(f"fft_bins: {frame_size // 2 + 1} (real FFT, including DC)")
    print(f"fft_frequency_resolution: {sample_rate / frame_size:.2f} Hz")


def plot(audio: np.ndarray, sample_rate: int, output: Path, frame_ms: float) -> None:
    import matplotlib.pyplot as plt

    time = np.arange(audio.size) / sample_rate
    frame_size = min(audio.size, max(2, round(sample_rate * frame_ms / 1000)))
    frame = audio[:frame_size]
    window = np.hanning(frame_size)
    windowed_frame = frame * window
    spectrum = np.abs(np.fft.rfft(windowed_frame))
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    frame_time = np.arange(frame_size) * 1000 / sample_rate

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(time, audio, linewidth=0.5)
    axes[0].set(title="1. Normalized waveform (time domain)", xlabel="Time (s)", ylabel="Amplitude")
    axes[1].plot(frame_time, frame, linewidth=0.8, label="waveform")
    axes[1].set(title=f"2. First {frame_ms:g} ms: signal and Hann window", xlabel="Time (ms)", ylabel="Waveform amplitude")
    window_axis = axes[1].twinx()
    window_axis.plot(frame_time, window, linewidth=1.5, color="tab:orange", label="Hann window")
    window_axis.set_ylabel("Window weight", color="tab:orange")
    window_axis.tick_params(axis="y", labelcolor="tab:orange")
    axes[1].legend(loc="upper left")
    window_axis.legend(loc="upper right")
    axes[2].plot(frame_time, windowed_frame, linewidth=0.8, color="tab:green")
    step = max(1, frame_size // 80)
    axes[2].scatter(frame_time[::step], windowed_frame[::step], s=8, color="tab:green")
    axes[2].set(title="3. Windowed samples (actual FFT input)", xlabel="Time (ms)", ylabel="Amplitude")
    axes[3].plot(frequencies, 20 * np.log10(np.maximum(spectrum, 1e-8)), linewidth=0.8)
    axes[3].set(title=f"4. FFT of one frame (bin spacing: {sample_rate / frame_size:.1f} Hz)", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
    axes[3].set_xlim(0, min(sample_rate / 2, 8000))
    figure.savefig(output, dpi=150)
    print(f"plot_saved: {output}")


def stft(audio: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return magnitude spectra, frame-center times, and frequency bins."""
    frame_size = max(2, round(sample_rate * frame_ms / 1000))
    hop_size = max(1, round(sample_rate * hop_ms / 1000))
    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))
    frame_count = 1 + (audio.size - frame_size) // hop_size
    window = np.hanning(frame_size)
    frames = np.stack(
        [audio[start : start + frame_size] * window for start in range(0, frame_count * hop_size, hop_size)]
    )
    spectra = np.abs(np.fft.rfft(frames, axis=1))
    times = (np.arange(frame_count) * hop_size + frame_size / 2) / sample_rate
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    return spectra, times, frequencies


def plot_stft(audio: np.ndarray, sample_rate: int, output: Path, frame_ms: float, hop_ms: float) -> None:
    import matplotlib.pyplot as plt

    spectra, times, frequencies = stft(audio, sample_rate, frame_ms, hop_ms)
    db = 20 * np.log10(np.maximum(spectra, 1e-8))
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = axis.pcolormesh(times, frequencies, db.T, shading="auto", cmap="magma")
    axis.set(title=f"STFT: {frame_ms:g} ms window, {hop_ms:g} ms hop ({spectra.shape[0]} frames)", xlabel="Time (s)", ylabel="Frequency (Hz)")
    axis.set_ylim(0, min(sample_rate / 2, 8000))
    figure.colorbar(image, ax=axis, label="Magnitude (dB)")
    figure.savefig(output, dpi=150)
    print(f"stft_shape: {spectra.shape} (frames, frequency_bins)")
    print(f"stft_plot_saved: {output}")


def animate_stft(audio: np.ndarray, sample_rate: int, output: Path, frame_ms: float, hop_ms: float, fps: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    spectra, times, frequencies = stft(audio, sample_rate, frame_ms, hop_ms)
    frame_size = max(2, round(sample_rate * frame_ms / 1000))
    hop_size = max(1, round(sample_rate * hop_ms / 1000))
    window = np.hanning(frame_size)
    db = 20 * np.log10(np.maximum(spectra, 1e-8))
    frame_indices = np.unique(np.linspace(0, len(times) - 1, min(len(times), 240), dtype=int))
    max_db = float(np.max(db))
    min_db = max_db - 80

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    time = np.arange(audio.size) / sample_rate
    axes[0].plot(time, audio, linewidth=0.5, color="0.35")
    axes[0].set(title="1. Sliding analysis window", xlabel="Time (s)", ylabel="Amplitude")
    window_start = axes[0].axvline(0, color="tab:orange", linewidth=2, label="window start")
    window_end = axes[0].axvline(frame_size / sample_rate, color="tab:red", linewidth=2, label="window end")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(0, time[-1])
    frame_axis = axes[1]
    frame_axis.set(title="2. Current frame and its spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
    spectrum_line, = frame_axis.plot([], [], color="tab:blue")
    frame_axis.set_xlim(0, min(sample_rate / 2, 8000))
    frame_axis.set_ylim(min_db, max_db + 3)
    image = axes[2].imshow(np.full_like(db.T, min_db), origin="lower", aspect="auto", extent=[times[0], times[-1], frequencies[0], frequencies[-1]], cmap="magma", vmin=min_db, vmax=max_db)
    axes[2].set(title="3. STFT columns accumulated over time", xlabel="Time (s)", ylabel="Frequency (Hz)")
    axes[2].set_ylim(0, min(sample_rate / 2, 8000))
    figure.colorbar(image, ax=axes[2], label="Magnitude (dB)")

    def update(position: int):
        index = int(frame_indices[position])
        window_start.set_xdata([times[index], times[index]])
        window_end.set_xdata([times[index] + frame_size / sample_rate, times[index] + frame_size / sample_rate])
        spectrum_line.set_data(frequencies, db[index])
        accumulated = np.full_like(db.T, min_db)
        accumulated[:, : index + 1] = db[: index + 1].T
        image.set_data(accumulated)
        axes[0].set_title(f"1. Sliding analysis window: frame {index + 1}/{len(times)} ({times[index]:.2f} s)")
        return window_start, window_end, spectrum_line, image

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=1000 / max(fps, 1), blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)
    print(f"stft_animation_frames: {len(frame_indices)}")
    print(f"stft_animation_saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="path to a 16-bit PCM WAV file")
    parser.add_argument("--plot", type=Path, help="optional output PNG path")
    parser.add_argument("--stft-plot", type=Path, help="optional STFT spectrogram PNG path")
    parser.add_argument("--stft-gif", type=Path, help="optional animated STFT GIF path")
    parser.add_argument("--frame-ms", type=float, default=25.0, help="analysis frame length for the teaching plot (default: 25 ms)")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="STFT hop length in milliseconds (default: 10 ms)")
    parser.add_argument("--fps", type=int, default=12, help="frames per second for --stft-gif (default: 12)")
    args = parser.parse_args()
    if args.frame_ms <= 0:
        parser.error("--frame-ms must be greater than 0")
    if args.hop_ms <= 0:
        parser.error("--hop-ms must be greater than 0")
    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if not args.audio.is_file():
        parser.error(
            f"audio file not found: {args.audio}\n"
            "Use a real WAV path; 'path/to/example.wav' is only a placeholder."
        )
    audio, sample_rate = read_wav(args.audio)
    describe(audio, sample_rate, args.frame_ms)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(audio, sample_rate, args.plot, args.frame_ms)
    if args.stft_plot:
        args.stft_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_stft(audio, sample_rate, args.stft_plot, args.frame_ms, args.hop_ms)
    if args.stft_gif:
        args.stft_gif.parent.mkdir(parents=True, exist_ok=True)
        animate_stft(audio, sample_rate, args.stft_gif, args.frame_ms, args.hop_ms, args.fps)


if __name__ == "__main__":
    main()
