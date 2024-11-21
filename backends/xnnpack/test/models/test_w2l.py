# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from torchaudio import models


class TestW2L(unittest.TestCase):
    batch_size = 10
    input_frames = 700
    vocab_size = 4096
    num_features = 1
    wav2letter = models.Wav2Letter(num_classes=vocab_size).eval()

    model_inputs = (torch.randn(batch_size, num_features, input_frames),)
    dynamic_shape = ({0: torch.export.Dim("batch", min=2, max=10)},)

    def test_fp32_w2l(self):
        (
            Tester(self.wav2letter, self.model_inputs, self.dynamic_shape)
            .export()
            .to_edge_transform_and_lower()
            .check_not(
                [
                    "executorch_exir_dialectes_edge__ops_aten_convolution_default",
                    "executorch_exir_dialects_edge__ops_aten_relu_default",
                ]
            )
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=10)
        )

    def test_qs8_w2l(self):
        (
            Tester(self.wav2letter.eval(), self.model_inputs, self.dynamic_shape)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_not(
                [
                    "executorch_exir_dialectes_edge__ops_aten_convolution_default",
                    "executorch_exir_dialects_edge__ops_aten_relu_default",
                ]
            )
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=10)
        )
