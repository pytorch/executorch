# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LFM3-VL image preprocessing and vision RoPE tables for a fixed square input.

Mirrors Lfm3VlImageProcessor (rescale, normalize, merge-block patchify with a
duplicated temporal frame) without the dynamic resize: images are resized to
`size` x `size`, which HF leaves untouched when `size` is a multiple of 32.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PATCH, MERGE, TEMPORAL = 16, 2, 2


def preprocess_image(path: str, size: int = 448) -> torch.Tensor:
    """Returns pixel_values [(size/16)**2, 3 * 2 * 16 * 16] in merge-block order."""
    return preprocess_pil(Image.open(path), size)


def preprocess_pil(img: Image.Image, size: int = 448) -> torch.Tensor:
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    c, h, w = x.shape
    gh, gw = h // PATCH, w // PATCH
    p = x.reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.permute(1, 4, 2, 5, 0, 3, 6)  # [gh/m, gw/m, m, m, c, P, P]
    p = p.unsqueeze(5).expand(-1, -1, -1, -1, -1, TEMPORAL, -1, -1)
    return p.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH).contiguous()


def vision_rope(size: int, head_dim: int, theta: float):
    """Axial 2D RoPE cos/sin [(size/16)**2, head_dim] in merge-block token order."""
    g = size // PATCH
    hb, wb, mh, mw = torch.meshgrid(
        torch.arange(g // MERGE),
        torch.arange(g // MERGE),
        torch.arange(MERGE),
        torch.arange(MERGE),
        indexing="ij",
    )
    hpos = (hb * MERGE + mh).reshape(-1).float()
    wpos = (wb * MERGE + mw).reshape(-1).float()
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2).float() / spatial))
    freqs = torch.cat([hpos[:, None] * inv_freq, wpos[:, None] * inv_freq], dim=-1)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


# Speech front end (matches the HF Lfm3OmniFeatureExtractor).
SAMPLE_RATE, N_FFT, HOP, N_MELS = 16000, 400, 160, 128


def load_wav(path: str) -> torch.Tensor:
    """Mono float waveform at 16 kHz."""
    import torchaudio
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    wav = torch.from_numpy(np.asarray(data, dtype=np.float32))
    if np.issubdtype(data.dtype, np.integer):
        wav = wav / float(np.iinfo(data.dtype).max + 1)
    if wav.ndim == 2:
        wav = wav.mean(1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def audio_features(wav: torch.Tensor, max_frames: int) -> torch.Tensor:
    """16 kHz waveform -> [1, N_MELS + 1, max_frames]: log-mel features and a 0/1
    row marking valid frames (the static encoder's input)."""
    from transformers.audio_utils import mel_filter_bank

    pad = N_FFT if wav.numel() < N_FFT else wav.numel() + (-(wav.numel() - N_FFT) % HOP)
    wav = F.pad(wav, (0, pad - wav.numel()))
    spec = torch.stft(
        wav,
        N_FFT,
        HOP,
        N_FFT,
        torch.hann_window(N_FFT),
        center=False,
        return_complex=True,
    )
    filters = torch.from_numpy(
        mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1,
            num_mel_filters=N_MELS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        )
    ).float()
    mel = ((filters.T @ spec.abs().square()).log10().clamp(min=-12.0) + 4.0) / 4.0
    frames = mel.shape[-1]
    if frames > max_frames:
        raise ValueError(f"{frames} mel frames exceed the encoder's {max_frames}")
    out = torch.zeros(1, N_MELS + 1, max_frames)
    out[0, :N_MELS, :frames] = mel
    out[0, N_MELS, :frames] = 1.0
    return out
