# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestConvertToLinear(unittest.TestCase):
    PassStage = RunPasses([ConvertToLinearPass])

    def test_fp32_convert_to_linear(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        bias_vals = [True, True, False]

        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=bias_vals[i])
            inputs = (torch.randn(in_size, input_size),)

            (
                Tester(linear, inputs)
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check_count(
                    {"executorch_exir_dialects_edge__ops_aten_linear_default": 1}
                )
                .run_method_and_compare_outputs()
            )
