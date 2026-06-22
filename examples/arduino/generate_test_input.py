#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate MFCC test input header from a .wav audio file.

Usage:
    python generate_test_input.py --input recording.wav --output mfcc_custom.h

The output .h file can be included in an Arduino sketch:
    #include "mfcc_custom.h"

Requirements:
    pip install soundfile numpy torch torchaudio
"""

import argparse

import numpy as np
import soundfile as sf
import torch


def wav_to_mfcc(path: str) -> np.ndarray:
    """Extract 49x10 MFCC features from a 1-second 16kHz audio file."""
    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data[:, 0]

    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio

        data = torch.from_numpy(data).unsqueeze(0)
        data = torchaudio.functional.resample(data, sr, 16000).squeeze(0).numpy()
        sr = 16000

    wav = torch.from_numpy(data).unsqueeze(0)

    # Pad or trim to exactly 1 second (16000 samples)
    if wav.shape[1] < 16000:
        wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[1]))
    else:
        wav = wav[:, :16000]

    # STFT
    n_fft, hop = 640, 320
    window = torch.hann_window(n_fft)
    spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
    power = spec.abs() ** 2

    # Mel filterbank (40 bands, 0-8kHz)
    n_mels = 40
    mel_pts = torch.linspace(
        2595 * np.log10(1 + 0 / 700), 2595 * np.log10(1 + 8000 / 700), n_mels + 2
    )
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = (hz_pts * n_fft / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            if bins[m + 1] > bins[m]:
                fb[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            if bins[m + 2] > bins[m + 1]:
                fb[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])

    mel_spec = torch.matmul(fb, power.squeeze(0))
    log_mel = torch.log(mel_spec + 1e-6)

    # DCT → 10 MFCCs
    n_mfcc = 10
    dct_mat = torch.zeros(n_mfcc, n_mels)
    for i in range(n_mfcc):
        for j in range(n_mels):
            dct_mat[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n_mels))

    mfcc = torch.matmul(dct_mat, log_mel)[:, :49]  # 49 time frames

    # Reshape to model input format: (1, 1, 49, 10), channels_last
    mfcc = mfcc.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
    return mfcc.contiguous().view(-1).numpy()


def main():
    parser = argparse.ArgumentParser(description="Generate MFCC header from .wav file")
    parser.add_argument("--input", required=True, help="Input .wav file (16kHz, 1 sec)")
    parser.add_argument("--output", required=True, help="Output .h file")
    parser.add_argument("--label", default="custom", help="Label for the audio")
    args = parser.parse_args()

    mfcc = wav_to_mfcc(args.input)

    h = f"// MFCC from: {args.input}\n"
    h += "// 49 time frames x 10 MFCC coefficients = 490 float values\n"
    h += "#pragma once\n\n"
    h += "static const float test_input[490] = {\n"
    for i in range(0, len(mfcc), 8):
        h += "    " + ",".join(f"{v:.6f}f" for v in mfcc[i : i + 8]) + ",\n"
    h += "};\n"
    h += f'static const char* test_label = "{args.label}";\n'

    with open(args.output, "w") as f:
        f.write(h)

    print(f"Generated {args.output} from {args.input} ({len(mfcc)} values)")


if __name__ == "__main__":
    main()
