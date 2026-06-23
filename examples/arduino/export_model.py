#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export DS-CNN keyword spotting model to model.h for Arduino.

Usage:
    # Train on Google Speech Commands and export:
    python export_model.py --output examples/KeywordSpotting/model.h

    # Export with a pre-trained checkpoint:
    python export_model.py --checkpoint my_weights.pth --output model.h

Requirements:
    pip install executorch soundfile numpy torch torchao
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.examples.models.mlperf_tiny.ds_cnn import DSCNNKWS
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def wav_to_mfcc(path: str) -> torch.Tensor:
    """Extract 49x10 MFCC features from a 1-second 16kHz audio file."""
    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data[:, 0]
    wav = torch.from_numpy(data).unsqueeze(0)
    if wav.shape[1] < 16000:
        wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[1]))
    else:
        wav = wav[:, :16000]
    n_fft, hop = 640, 320
    window = torch.hann_window(n_fft)
    spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
    power = spec.abs() ** 2
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
    dct_mat = torch.zeros(10, n_mels)
    for i in range(10):
        for j in range(n_mels):
            dct_mat[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n_mels))
    mfcc = torch.matmul(dct_mat, log_mel)[:, :49]
    return (
        mfcc.unsqueeze(0)
        .unsqueeze(0)
        .permute(0, 1, 3, 2)
        .to(memory_format=torch.channels_last)
    )


def train_model(data_dir: str, samples_per_class: int = 100) -> DSCNNKWS:
    """Train DS-CNN on 10 keyword classes from Google Speech Commands.

    Indices 0 (silence) and 1 (unknown) are reserved to match the 12-class
    MLPerf label mapping but are not trained here.
    """
    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    train_x, train_y = [], []

    print("Loading training data...")
    for cls_idx, name in enumerate(labels, start=2):
        wavs = sorted(
            f for f in os.listdir(os.path.join(data_dir, name)) if f.endswith(".wav")
        )[:samples_per_class]
        for wav in wavs:
            train_x.append(wav_to_mfcc(os.path.join(data_dir, name, wav)))
            train_y.append(cls_idx)
        print(f"  {name}: {len(wavs)}")

    train_x = torch.cat(train_x)
    train_y = torch.tensor(train_y)
    print(f"Total: {len(train_y)} samples")

    model = DSCNNKWS().train()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(80):
        idx = torch.randperm(len(train_y))
        correct = 0
        for i in range(0, len(idx), 32):
            b = idx[i : i + 32]
            opt.zero_grad()
            out = model(train_x[b])
            loss = crit(out, train_y[b])
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == train_y[b]).sum().item()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: acc={correct / len(train_y):.0%}")

    return model.eval()


def export_model(
    model: DSCNNKWS, calibration_dir: str, target: str = "cortex-m33"
) -> bytes:
    """Quantize and export model to .pte bytes."""
    example = (torch.rand(1, 1, 49, 10) * 2 - 1).to(memory_format=torch.channels_last)

    exported_program = export(model, (example,), strict=True)
    captured = exported_program.module(check_guards=False)

    # Calibrate with real audio
    quantizer = CortexMQuantizer()
    prepared = prepare_pt2e(captured, quantizer)

    print("Calibrating with real audio...")
    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    for label in labels:
        label_dir = os.path.join(calibration_dir, label)
        wavs = sorted(f for f in os.listdir(label_dir) if f.endswith(".wav"))
        for wav in wavs[50:55]:
            prepared(wav_to_mfcc(os.path.join(label_dir, wav)))

    converted = convert_pt2e(prepared, fold_quantize=True)
    DuplicateDynamicQuantChainPass()(converted)

    # Export with Cortex-M passes
    cpu = CortexM.M33 if "m33" in target else CortexM.M55
    edge = to_edge(
        export(converted, (example,)),
        compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.linear.default],
            _check_ir_validity=False,
            _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
        ),
    )
    pm = CortexMPassManager(
        edge.exported_program(),
        CortexMPassManager.pass_list,
        target_config=CortexMTargetConfig(cpu=cpu),
    )
    edge._edge_programs["forward"] = pm.transform()
    et = edge.to_executorch()
    return et.buffer


def buffer_to_header(buffer: bytes) -> str:
    """Convert .pte bytes to a C header string."""
    h = "#pragma once\n#include <cstdint>\n#include <cstddef>\n\n"
    h += "alignas(16) static const uint8_t model_pte[] = {\n"
    for i in range(0, len(buffer), 16):
        h += "    " + ",".join(f"0x{b:02x}" for b in buffer[i : i + 16]) + ",\n"
    h += "};\n"
    h += f"static const size_t model_pte_size = {len(buffer)};\n"
    return h


def main():
    parser = argparse.ArgumentParser(
        description="Export DS-CNN keyword spotting model for Arduino"
    )
    parser.add_argument("--output", required=True, help="Output model.h path")
    parser.add_argument("--checkpoint", help="Pre-trained .pth weights (skip training)")
    parser.add_argument(
        "--data-dir",
        default="outputs/speech_commands/SpeechCommands/speech_commands_v0.02",
        help="Google Speech Commands dataset directory",
    )
    parser.add_argument(
        "--target", default="cortex-m33", help="Target CPU (cortex-m33 or cortex-m55)"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Training samples per class"
    )
    args = parser.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = DSCNNKWS().eval()
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        if not os.path.isdir(args.data_dir):
            print(f"Dataset not found at {args.data_dir}")
            print(
                'Download: python -c "import torchaudio; '
                "torchaudio.datasets.SPEECHCOMMANDS(root='outputs/speech_commands', "
                'download=True)"'
            )
            return
        model = train_model(args.data_dir, args.samples)
        torch.save(model.state_dict(), args.output.replace(".h", ".pth"))

    buffer = export_model(model, args.data_dir, args.target)
    header = buffer_to_header(buffer)

    with open(args.output, "w") as f:
        f.write(header)
    print(f"\n{args.output}: {len(buffer)} bytes ({len(buffer) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
