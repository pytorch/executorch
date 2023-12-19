# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import TestXNNPACK

from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    weight_observer_range_neg_127_to_127,
)


class TestXNNPACKQuantized(TestXNNPACK):
    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.elu = torch.nn.ELU(alpha=0.5)

            def forward(self, x):
                return self.elu(x)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu2(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.elu(x, alpha=1.2)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)

    def test_xnnpack_dqlinear_mm_per_tensor(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=False
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_addmm_per_tensor(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=True
        )

    def test_xnnpack_dqlinear_mm_per_channel(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=False,
        )

    def test_xnnpack_dqlinear_addmm_per_channel(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=True,
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_partitioner_mm_per_tensor(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=False
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_partitioner_addmm_per_tensor(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=True
        )

    def test_xnnpack_dqlinear_partitioner_mm_per_channel(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=False,
        )

    def test_xnnpack_dqlinear_partitioner_addmm_per_channel(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=True,
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_parallel(self):
        use_bias = True

        in_size = 1
        input_size = 4
        output_size = 5

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                return (a, b)

        example_inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_sequential(self):
        use_bias = True

        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    a, self.linear2_weight, self.linear2_bias
                )
                return b

        example_inputs = (torch.rand(in_size, input_size, dtype=torch.float),)

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_parallel_and_sequential(self):
        use_bias = True

        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear3_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear3_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                c = torch.nn.functional.linear(
                    b, self.linear3_weight, self.linear3_bias
                )
                return (a, c)

        example_inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )
