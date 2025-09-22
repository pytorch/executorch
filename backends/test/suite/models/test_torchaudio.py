# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Tuple

import pytest
import torch
import torchaudio

from executorch.backends.test.suite import dtype_to_str
from executorch.backends.test.suite.flow import TestFlow
from torch.export import Dim

#
# This file contains model integration tests for supported torchaudio models. As many torchaudio
# models are not export-compatible, this suite contains a subset of the available models and may
# grow over time.
#


class PatchedConformer(torch.nn.Module):
    """
    A lightly modified version of the top-level Conformer module, such that it can be exported.
    Instead of taking lengths and computing the padding mask, it takes the padding mask directly.
    See https://github.com/pytorch/audio/blob/main/src/torchaudio/models/conformer.py#L215
    """

    def __init__(self, conformer):
        super().__init__()
        self.conformer = conformer

    def forward(
        self, input: torch.Tensor, encoder_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = input.transpose(0, 1)
        for layer in self.conformer.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float32], ids=dtype_to_str)
@pytest.mark.parametrize(
    "use_dynamic_shapes", [False, True], ids=["static_shapes", "dynamic_shapes"]
)
def test_conformer(test_runner, dtype: torch.dtype, use_dynamic_shapes: bool):
    inner_model = torchaudio.models.Conformer(
        input_dim=80,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        depthwise_conv_kernel_size=31,
    )
    model = PatchedConformer(inner_model).eval().to(dtype)
    lengths = torch.randint(1, 400, (10,))

    encoder_padding_mask = torchaudio.models.conformer._lengths_to_padding_mask(lengths)
    inputs = (
        torch.rand(10, int(lengths.max()), 80),
        encoder_padding_mask,
    )

    test_runner.lower_and_run_model(model, inputs)


@pytest.mark.parametrize("dtype", [torch.float32], ids=dtype_to_str)
@pytest.mark.parametrize(
    "use_dynamic_shapes", [False, True], ids=["static_shapes", "dynamic_shapes"]
)
def test_wav2letter(flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool):
    model = torchaudio.models.Wav2Letter().to(dtype)
    inputs = (torch.randn(1, 1, 1024, dtype=dtype),)
    dynamic_shapes = (
        {
            "x": {
                2: Dim("d", min=900, max=1024),
            }
        }
        if use_dynamic_shapes
        else None
    )

    test_runner.lower_and_run_model(model, inputs)


@pytest.mark.parametrize("dtype", [torch.float32], ids=dtype_to_str)
@pytest.mark.parametrize(
    "use_dynamic_shapes", [False, True], ids=["static_shapes", "dynamic_shapes"]
)
@unittest.skip("This model times out on all backends.")
def test_wavernn(
    test_runner,
    dtype: torch.dtype,
    use_dynamic_shapes: bool,
):
    model = (
        torchaudio.models.WaveRNN(
            upsample_scales=[5, 5, 8], n_classes=512, hop_length=200
        )
        .eval()
        .to(dtype)
    )

    # See https://docs.pytorch.org/audio/stable/generated/torchaudio.models.WaveRNN.html#forward
    inputs = (
        torch.randn(1, 1, (64 - 5 + 1) * 200),  # waveform
        torch.randn(1, 1, 128, 64),  # specgram
    ).to(dtype)

    test_runner.lower_and_run_model(model, inputs)
