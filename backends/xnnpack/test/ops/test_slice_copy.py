# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSliceCopy(unittest.TestCase):
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
        self._test_slice_copy(ConvSlice(), inputs, 4, 2)

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
            .check_count({"torch.ops.aten.slice.Tensor": 3})
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

    # Note: Slice ends up as slice_copy later in the process, but during quantization,
    # it's still slice, which isn't supported by the XNNPACK quantizer.
    @unittest.skip("T156004676 - slice isn't propagated")
    def _test_qs8_slice_copy(self):
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
                    "aten::slice.Tensor": 3,
                    "quantized_decomposed::quantize_per_tensor": 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
