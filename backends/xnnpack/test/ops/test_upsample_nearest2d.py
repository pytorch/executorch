# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.config.generic_node_configs import (
    UpsampleNearest2dConfig,
)
from executorch.backends.xnnpack.test.tester import Tester


class TestUpsampleNearest2d(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Nearest2x(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

    class Nearest2xBySize(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, size=(x.shape[2] * 2, x.shape[3] * 2), mode="nearest"
            )

    class Nearest3x(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=3.0, mode="nearest")

    class NearestInexact2x(torch.nn.Module):
        """Doubles the size, but PyTorch indexes with 1/2.004, so the x2
        duplication lowering would not match eager."""

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=2.004, mode="nearest"
            )

    class ConvNearestConv(torch.nn.Module):
        """The pattern that used to split XNNPACK partitions in conv decoders."""

        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(8, 4, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            return self.conv2(x)

    edge_op = "executorch_exir_dialects_edge__ops_aten_upsample_nearest2d_vec"

    def _test_lowered(self, module, inputs, quant=False):
        tester = Tester(module, inputs)
        if quant:
            tester.quantize()
        tester.export()
        if quant:
            tester.check(["torch.ops.quantized_decomposed"])
        (
            tester.to_edge_transform_and_lower()
            .check_not([self.edge_op])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=2 if quant else 0)
        )

    def test_fp32_upsample_nearest2d_scale_factor(self):
        self._test_lowered(self.Nearest2x(), (torch.randn(2, 3, 4, 5),))

    def test_fp32_upsample_nearest2d_size(self):
        self._test_lowered(self.Nearest2xBySize(), (torch.randn(1, 5, 7, 3),))

    def test_fp16_upsample_nearest2d(self):
        self._test_lowered(
            self.Nearest2x(), (torch.randn(1, 6, 8, 8).to(torch.float16),)
        )

    def test_fp32_conv_upsample_conv_single_partition(self):
        self._test_lowered(self.ConvNearestConv(), (torch.randn(1, 3, 16, 16),))

    def test_qs8_conv_upsample_conv_single_partition(self):
        self._test_lowered(
            self.ConvNearestConv(), (torch.randn(1, 3, 16, 16),), quant=True
        )

    def test_qs8_upsample_nearest2d_requantized_output(self):
        """The output is quantized differently from the input, so the
        delegate requantizes it after depth_to_space."""

        class RequantizedNearest2x(torch.nn.Module):
            def forward(self, x):
                quantized = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.25, 0, -128, 127, torch.int8
                )
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                        quantized, 0.25, 0, -128, 127, torch.int8
                    )
                )
                upsampled = torch.nn.functional.interpolate(
                    dequantized, scale_factor=2.0, mode="nearest"
                )
                requantized = (
                    torch.ops.quantized_decomposed.quantize_per_tensor.default(
                        upsampled, 0.5, 0, -128, 127, torch.int8
                    )
                )
                return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    requantized, 0.5, 0, -128, 127, torch.int8
                )

        (
            Tester(RequantizedNearest2x(), (torch.randn(1, 3, 4, 4),))
            .export()
            .to_edge_transform_and_lower()
            .check_not([self.edge_op])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=1)
        )

    def test_fp32_upsample_nearest2d_x3_not_partitioned(self):
        (
            Tester(self.Nearest3x(), (torch.randn(1, 3, 4, 4),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({self.edge_op: 1})
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    def test_fp32_upsample_nearest2d_inexact_scale_not_partitioned(self):
        (
            Tester(self.NearestInexact2x(), (torch.randn(1, 3, 200, 200),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({self.edge_op: 1})
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_upsample_nearest2d_unsupported_scale_not_annotated(self):
        # The quantizer only annotates what the partitioner lowers, so an
        # upsample that stays outside the delegate is not wrapped in q/dq.
        for module, inputs in (
            (self.Nearest3x(), (torch.randn(1, 3, 4, 4),)),
            (self.NearestInexact2x(), (torch.randn(1, 3, 200, 200),)),
        ):
            (
                Tester(module, inputs)
                .quantize()
                .export()
                .check_not(["torch.ops.quantized_decomposed"])
                .to_edge_transform_and_lower()
                .check_count({self.edge_op: 1})
                .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            )

    def test_fp32_upsample_nearest2d_inexact_scale_kwarg_rejected(self):
        # torch.export passes scale_factors positionally; a graph built by
        # hand or rewritten by a pass can carry it as a kwarg instead.
        ep = torch.export.export(
            self.NearestInexact2x(), (torch.randn(1, 3, 200, 200),)
        )
        node = next(
            n
            for n in ep.graph.nodes
            if n.target == torch.ops.aten.upsample_nearest2d.vec
        )
        node.kwargs = {"scale_factors": node.args[2]}
        node.args = node.args[:2]
        self.assertFalse(UpsampleNearest2dConfig().check_constraints(node, ep))
