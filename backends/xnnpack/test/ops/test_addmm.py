# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester, ToEdgeTransformAndLower


class TestAddmm(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class AddmmWithScales(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))
            self.bias = torch.nn.Parameter(torch.randn(8))

        def forward(self, x):
            return (
                # partitioned
                torch.addmm(self.bias, x, self.weight),
                # not partitioned, since beta or alpha is not 1
                torch.addmm(self.bias, x, self.weight, beta=2.0),
                torch.addmm(self.bias, x, self.weight, alpha=0.5),
            )

    def test_fp32_addmm_with_scales(self):
        (
            Tester(self.AddmmWithScales(), (torch.randn(6, 16),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_addmm_default": 2})
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class AddmmWithBiases(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))
            self.channel_bias = torch.nn.Parameter(torch.randn(8))
            self.row_bias = torch.nn.Parameter(torch.randn(1, 8))
            self.full_bias = torch.nn.Parameter(torch.randn(8, 8))
            self.column_bias = torch.nn.Parameter(torch.randn(8, 1))
            self.scalar_bias = torch.nn.Parameter(torch.randn(1))
            self.one_channel_weight = torch.nn.Parameter(torch.randn(16, 1))
            self.one_channel_bias = torch.nn.Parameter(torch.randn(()))

        def forward(self, x):
            return (
                # partitioned, since the bias has 1 value per output channel
                torch.addmm(self.channel_bias, x, self.weight),
                torch.addmm(self.row_bias, x, self.weight),
                torch.addmm(self.one_channel_bias, x, self.one_channel_weight),
                # not partitioned
                torch.addmm(self.full_bias, x, self.weight),
                torch.addmm(self.column_bias, x, self.weight),
                torch.addmm(self.scalar_bias, x, self.weight),
            )

    def test_fp32_addmm_with_biases(self):
        (
            Tester(self.AddmmWithBiases(), (torch.randn(8, 16),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_addmm_default": 3})
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class LinearWithBiases(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 16))
            self.channel_bias = torch.nn.Parameter(torch.randn(8))
            self.row_bias = torch.nn.Parameter(torch.randn(1, 8))
            self.full_bias = torch.nn.Parameter(torch.randn(8, 8))
            self.column_bias = torch.nn.Parameter(torch.randn(8, 1))
            self.scalar_bias = torch.nn.Parameter(torch.randn(1))

        def forward(self, x):
            linear = torch.nn.functional.linear
            return (
                # partitioned, since the bias has 1 value per output channel
                linear(x, self.weight, self.channel_bias),
                linear(x, self.weight, self.row_bias),
                # not partitioned
                linear(x, self.weight, self.full_bias),
                linear(x, self.weight, self.column_bias),
                linear(x, self.weight, self.scalar_bias),
            )

    def test_fp32_decomposed_linear_with_biases(self):
        # to_edge decomposes F.linear into addmm(bias, x, weight.t()), which
        # XNNPACK takes through the linear source partition.
        (
            Tester(self.LinearWithBiases(), (torch.randn(8, 16),))
            .export()
            .to_edge()
            .partition()
            .check_count({"executorch_exir_dialects_edge__ops_aten_addmm_default": 3})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class AddmmWithRuntimeBiases(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))

        def forward(self, channel_bias, full_bias, x, n):
            k = n.item()
            torch._check(k >= 0)
            torch._check(k <= x.shape[0])
            return (
                # partitioned, since the bias has 1 value per output channel
                torch.addmm(channel_bias, x, self.weight),
                # not partitioned, since the number of bias rows is dynamic
                torch.addmm(full_bias, x, self.weight),
                # not partitioned, and checking a bias whose size depends on the
                # data must not fail the lowering
                torch.addmm(full_bias.narrow(0, 0, k), x.narrow(0, 0, k), self.weight),
            )

    def test_fp32_addmm_with_runtime_biases(self):
        # This flag lets a bias that is a program input into the delegate.
        partitioner = XnnpackPartitioner(force_non_static_weights_for_f32_linear=True)
        inputs = (
            torch.randn(8),
            torch.randn(8, 8),
            torch.randn(8, 16),
            torch.tensor(3),
        )
        rows = torch.export.Dim("rows", max=16)
        (
            Tester(
                self.AddmmWithRuntimeBiases(),
                inputs,
                dynamic_shapes=(None, {0: rows}, {0: rows}, None),
            )
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(partitioners=[partitioner])
            )
            .check_count({"executorch_exir_dialects_edge__ops_aten_addmm_default": 2})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))

        def forward(self, x):
            return torch.mm(x, self.weight)

    def test_fp32_mm(self):
        (
            Tester(self.MM(), (torch.randn(8, 16),))
            .export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
