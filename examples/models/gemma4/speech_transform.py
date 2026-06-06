# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Speech transform module for Gemma 4.

Converts raw audio waveforms to log-mel spectrograms for the Gemma 4
audio encoder. No learned weights — pure signal processing.

Usage:
    from executorch.examples.models.gemma4.speech_transform import (
        Gemma4SpeechTransformModel,
    )

    model = Gemma4SpeechTransformModel()
    waveform = torch.randn(16000)  # 1 second at 16kHz
    mel_spec = model(waveform)  # [num_frames, 128]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mel_filterbank(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    fft_length: int,
) -> torch.Tensor:
    """Create a triangular mel filterbank matrix of shape [n_freqs, n_mels]."""
    all_freqs = torch.arange(n_freqs, dtype=torch.float32) * (sample_rate / fft_length)

    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)

    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]

    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]

    fb = torch.maximum(torch.zeros(1), torch.minimum(down_slopes, up_slopes))
    return fb


class Gemma4SpeechTransformModel(nn.Module):
    """
    Speech transform for Gemma 4.

    Converts raw waveforms to log-mel spectrograms. Unlike Gemma 3N, this uses:
    - 20ms frames (320 samples) with 10ms hop (160 samples)
    - No FFT overdrive (fft_length=512)
    - No preemphasis
    - Mel range 0-8000 Hz
    - mel_floor=0.001
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 320,
        hop_length: int = 160,
        fft_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        mel_floor: float = 0.001,
        input_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.mel_floor = mel_floor
        self.input_scale_factor = input_scale_factor

        window = torch.hann_window(frame_length, periodic=True)
        self.register_buffer("window", window)

        mel_filters = create_mel_filterbank(
            n_freqs=fft_length // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            fft_length=fft_length,
        )
        self.register_buffer("mel_filters", mel_filters)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.

        Matches HuggingFace Gemma4AudioFeatureExtractor._extract_spectrogram():
        - Semicausal time padding (prepend frame_length//2 zeros)
        - Frame size = frame_length + 1 (for preemphasis support), drop last sample
        - log(mel + mel_floor) for log compression

        Args:
            waveform: Shape [num_samples] (1D only).
                Caller should pad to multiple of 128 first if matching HF exactly.

        Returns:
            Log-mel spectrogram of shape [num_frames, n_mels]
        """
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        # Semicausal padding: prepend frame_length//2 zeros so first frame
        # is centered at t=0, matching HF's time_padding='semicausal'
        pad_left = self.frame_length // 2
        waveform = F.pad(waveform, (pad_left, 0))

        # Unfold with frame_length + 1 (extra sample for preemphasis support)
        frame_size = self.frame_length + 1
        frames = waveform.unfold(0, frame_size, self.hop_length)

        # Drop last sample (preemphasis=0 path: frames[..., :-1])
        frames = frames[..., :-1]

        # Apply Hann window
        frames = frames * self.window

        # FFT -> magnitude spectrum (rfft implicitly pads to fft_length)
        stft = torch.fft.rfft(frames, n=self.fft_length, dim=-1)
        magnitude = torch.abs(stft)

        # Mel filterbank
        mel_spec = torch.matmul(magnitude, self.mel_filters)

        # Log compression: log(mel + mel_floor) matches HF exactly
        log_mel_spec = torch.log(mel_spec + self.mel_floor)

        return log_mel_spec
