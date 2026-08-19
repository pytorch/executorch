# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.utils.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._fake_program import get_fake_program


class TestSliceCopy(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _test_slice_copy(self, module, inputs, copy_count=1, edge_copy_count=1):
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.slice.Tensor": copy_count})
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": edge_copy_count
                }
            )
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_slice_copy(self):
        class SliceCopy(torch.nn.Module):
            def forward(self, x):
                return x[1:3, -2:, :-1]

        inputs = (torch.randn(5, 5, 5).to(torch.float16),)
        self._test_slice_copy(SliceCopy(), inputs, 3, 3)

    def test_fp32_slice_copy(self):
        class SliceCopy(torch.nn.Module):
            def forward(self, x):
                return x[1:3, -2:, :-1]

        inputs = (torch.randn(5, 5, 5),)
        self._test_slice_copy(SliceCopy(), inputs, 3, 3)

    def test_fp32_slice_copy_memory_format(self):
        class ConvSlice(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x):
                y = self.conv(x)
                return y[:, :, 2:3, -2:]

        inputs = (torch.randn(1, 1, 3, 3),)
        # Note that two of the slices are optimized away as they are identity.
        self._test_slice_copy(ConvSlice(), inputs, 2, 2)

    def test_fp32_slice_copy_default_start(self):
        """
        XNNPACK supports default start in slice op.
        """

        class Slice(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.slice.Tensor(x, 0, None, 2)

        inputs = (torch.randn(5, 5),)
        self._test_slice_copy(Slice(), inputs, 1, 1)

    def test_fp32_slice_copy_stride_non_1(self):
        """
        XNNPACK does not support strided slicing.
        """

        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[:3:2, :, :]

        module = Slice()
        inputs = (torch.randn(5, 5, 5),)
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.slice.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    def test_fp32_slice_copy_dim_0(self):
        """
        XNNPACK does not support 0-size dims.
        """

        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[-1:3, 2:, 3:3]

        module = Slice()
        inputs = (torch.randn(5, 5, 5),)
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.slice.Tensor": 3})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    def test_fp32_static_slice_with_dynamic_dim(self):
        """
        XNNPACK does not support dynamic dims with static slice
        """

        class SliceCopy(torch.nn.Module):
            def forward(self, x):
                return x[1:3, -2:, :-1]

        inputs = (torch.randn(5, 5, 5),)
        (
            Tester(
                SliceCopy(),
                inputs,
                dynamic_shapes=({2: torch.export.Dim("dim_2", min=4, max=100)},),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    # Note: Slice ends up as slice_copy later in the process, but during quantization
    # it's still slice, so the quantizer shares observers on aten.slice.Tensor.
    def test_qs8_slice_copy(self):
        class SliceCopy(torch.nn.Module):
            def forward(self, x):
                y = x + x
                z = y[1:3, -2:, :-1]
                return z

        inputs = (torch.randn(5, 5, 5),)
        (
            Tester(SliceCopy(), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    torch.ops.aten.slice.Tensor: 3,
                    # 3 slices, plus the input and the add output
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_slice_copy_mismatched_qparams_falls_back(self):
        class SliceCopy(torch.nn.Module):
            def forward(self, x):
                quantized = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.25, 0, -128, 127, torch.int8
                )
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                        quantized, 0.25, 0, -128, 127, torch.int8
                    )
                )
                sliced = torch.ops.aten.slice.Tensor(dequantized, 1, 0, 2)
                requantized = (
                    torch.ops.quantized_decomposed.quantize_per_tensor.default(
                        sliced, 0.5, 0, -128, 127, torch.int8
                    )
                )
                return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    requantized, 0.5, 0, -128, 127, torch.int8
                )

        inputs = (torch.randn(1, 4, 3),)
        (
            Tester(SliceCopy(), inputs)
            .export()
            .to_edge()
            .partition()
            .check(["executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_per_channel_slice_copy_mismatched_qparams_falls_back(self):
        class SliceCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "input_scales", torch.tensor([0.25, 0.5, 0.75, 1.0])
                )
                self.register_buffer(
                    "output_scales", torch.tensor([0.5, 0.75, 1.0, 1.25])
                )
                self.register_buffer("zero_points", torch.zeros(4, dtype=torch.int64))

            def forward(self, x):
                quantized = torch.ops.quantized_decomposed.quantize_per_channel.default(
                    x, self.input_scales, self.zero_points, 1, -128, 127, torch.int8
                )
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_channel.default(
                        quantized,
                        self.input_scales,
                        self.zero_points,
                        1,
                        -128,
                        127,
                        torch.int8,
                    )
                )
                sliced = torch.ops.aten.slice.Tensor(dequantized, 2, 0, 2)
                requantized = (
                    torch.ops.quantized_decomposed.quantize_per_channel.default(
                        sliced,
                        self.output_scales,
                        self.zero_points,
                        1,
                        -128,
                        127,
                        torch.int8,
                    )
                )
                return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    requantized,
                    self.output_scales,
                    self.zero_points,
                    1,
                    -128,
                    127,
                    torch.int8,
                )

        inputs = (torch.randn(1, 4, 3),)
        (
            Tester(SliceCopy(), inputs)
            .export()
            .to_edge()
            .partition()
            .check(["executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"])
        )

    def test_qs8_per_channel_slice_copy_matching_distinct_constant_qparams(self):
        class SliceCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scales = torch.tensor([0.25, 0.5, 0.75, 1.0])
                self.output_scales = torch.tensor([0.25, 0.5, 0.75, 1.0])
                self.zero_points = torch.zeros(4, dtype=torch.int64)

            def forward(self, x):
                quantized = torch.ops.quantized_decomposed.quantize_per_channel.default(
                    x, self.input_scales, self.zero_points, 1, -128, 127, torch.int8
                )
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_channel.default(
                        quantized,
                        self.input_scales,
                        self.zero_points,
                        1,
                        -128,
                        127,
                        torch.int8,
                    )
                )
                sliced = torch.ops.aten.slice.Tensor(dequantized, 2, 0, 2)
                requantized = (
                    torch.ops.quantized_decomposed.quantize_per_channel.default(
                        sliced,
                        self.output_scales,
                        self.zero_points,
                        1,
                        -128,
                        127,
                        torch.int8,
                    )
                )
                return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    requantized,
                    self.output_scales,
                    self.zero_points,
                    1,
                    -128,
                    127,
                    torch.int8,
                )

        real_edge_program = (
            Tester(SliceCopy(), (torch.randn(1, 4, 3),))
            .export()
            .to_edge()
            .get_artifact()
            .exported_program()
        )
        edge_program = get_fake_program(real_edge_program)
        slice_node = next(
            node
            for node in edge_program.graph.nodes
            if node.target == exir_ops.edge.aten.slice_copy.Tensor
        )
        dequant_node = slice_node.args[0]
        quant_node = next(iter(slice_node.users))
        input_scales = get_param_tensor(edge_program, dequant_node.args[1])
        output_scales = get_param_tensor(edge_program, quant_node.args[1])

        self.assertIsNot(dequant_node.args[1], quant_node.args[1])
        if input_scales is None or output_scales is None:
            self.fail("Expected lifted scale constants")
        self.assertTrue(torch.equal(input_scales, output_scales))

        partition_result = XnnpackPartitioner().partition(edge_program)
        delegation_tag = slice_node.meta.get("delegation_tag")
        self.assertIsNotNone(delegation_tag)
        self.assertIn(delegation_tag, partition_result.partition_tags)
