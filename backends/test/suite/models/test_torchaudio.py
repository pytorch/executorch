# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Callable, Tuple

import torch
import torchaudio

from executorch.backends.test.suite.models import (
    model_test_cls,
    model_test_params,
    run_model_test,
)
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


@model_test_cls
class TorchAudio(unittest.TestCase):
    @model_test_params(dtypes=[torch.float32], supports_dynamic_shapes=False)
    def test_conformer(
        self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable
    ):
        inner_model = torchaudio.models.Conformer(
            input_dim=80,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )
        model = PatchedConformer(inner_model)
        lengths = torch.randint(1, 400, (10,))

        encoder_padding_mask = torchaudio.models.conformer._lengths_to_padding_mask(
            lengths
        )
        inputs = (
            torch.rand(10, int(lengths.max()), 80),
            encoder_padding_mask,
        )

        run_model_test(model, inputs, dtype, None, tester_factory)

    @model_test_params(dtypes=[torch.float32])
    def test_wav2letter(
        self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable
    ):
        model = torchaudio.models.Wav2Letter()
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
        run_model_test(model, inputs, dtype, dynamic_shapes, tester_factory)

    @unittest.skip("This model times out on all backends.")
    def test_wavernn(
        self, dtype: torch.dtype, use_dynamic_shapes: bool, tester_factory: Callable
    ):
        model = torchaudio.models.WaveRNN(
            upsample_scales=[5, 5, 8], n_classes=512, hop_length=200
        ).eval()

        # See https://docs.pytorch.org/audio/stable/generated/torchaudio.models.WaveRNN.html#forward
        inputs = (
            torch.randn(1, 1, (64 - 5 + 1) * 200),  # waveform
            torch.randn(1, 1, 128, 64),  # specgram
        )

        run_model_test(model, inputs, dtype, None, tester_factory)
