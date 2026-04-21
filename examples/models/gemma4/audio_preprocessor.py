"""Gemma4 audio preprocessor: raw PCM → log-mel spectrogram.

Mirrors the Gemma4AudioFeatureExtractor settings:
  sampling_rate: 16000 Hz
  fft_length:    512
  hop_length:    160
  feature_size:  128 (n_mels)
  mel_floor:     0.001
  preemphasis:   0.0
  input_scale_factor: 1.0

Reuses the torch.stft-based pattern from
  extension/audio/mel_spectrogram.py (WhisperAudioProcessor).

The module is exportable via torch.export with dynamic waveform length.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Return mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    freq_bins = torch.linspace(0, sample_rate / 2.0, n_freqs)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(m.item()) for m in mel_points])

    # For each mel band, build a triangular filter over freq bins
    filters = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        for j, f in enumerate(freq_bins):
            if left <= f <= center:
                filters[i, j] = (f - left) / (center - left + 1e-9)
            elif center < f <= right:
                filters[i, j] = (right - f) / (right - center + 1e-9)
    return filters


class Gemma4AudioPreprocessor(nn.Module):
    """Raw float PCM waveform → log-mel spectrogram.

    Input:  waveform  (1, N_samples) float32  — mono 16 kHz PCM
    Output: mel_spec  (1, T_frames, n_mels)   — time-major log-mel features

    This matches the input format expected by Gemma4AudioModel.

    torch.export compatible: STFT with static window size, static n_mels.
    T_frames is dynamic (Dim-constrained on export).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 128,
        mel_floor: float = 0.001,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_floor = mel_floor

        # Hann window — static buffer, not a parameter
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # Mel filterbank — static (n_mels, n_freqs) matrix
        mel_fb = _build_mel_filters(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_filters", mel_fb, persistent=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, N_samples)
        x = waveform.squeeze(0)  # (N_samples,)

        # STFT → complex spectrogram (n_freqs, T_frames) complex
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (n_freqs, T_frames)

        # Power spectrogram
        power = spec.real.pow(2) + spec.imag.pow(2)  # (n_freqs, T_frames)

        # Apply mel filterbank: (n_mels, n_freqs) @ (n_freqs, T) → (n_mels, T)
        mel = torch.matmul(self.mel_filters.to(power.dtype), power)

        # Floor and log
        mel = torch.clamp(mel, min=self.mel_floor)
        log_mel = torch.log(mel)

        # (n_mels, T) → (T, n_mels) → (1, T, n_mels)
        return log_mel.T.unsqueeze(0)
