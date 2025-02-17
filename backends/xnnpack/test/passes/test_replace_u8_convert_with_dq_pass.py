# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.replace_u8_convert_with_dq_pass import (
    ReplaceU8ConvertWithDqPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass


class TestReplaceU8ConvertWithDqPass(unittest.TestCase):
    PassStage = RunPasses(
        [
            DimOrderOpsRevertPass,
            ReplaceU8ConvertWithDqPass,
        ]
    )

    def test_single_op_convert(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .check_node_count(
                {
                    exir_ops.edge.aten._to_copy.default: 0,
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                }
            )
            .run_method_and_compare_outputs()
        )

    # TODO(gjcomer) Switch these tests to check nodes direct after removing the default decomposition for
    # upsample_bilinear2d in PyTorch. Because of the way to_edge works, we can't
    # easily enable these tests without it, since we can't opt out of the decomp
    # without using to_edge_transform_and_lower. The recompose pass for blinear
    # is also only written for f32 and it's throwaway work to enable u8.
    def test_convert_with_nchw_before(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = torch.nn.functional.interpolate(x, mode="bilinear", scale_factor=2)
                return y.to(torch.float)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs()
        )

        """
        .check_node_count({
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 1,
        })
        .run_passes(self.PassStage)
        .check_node_count({
            exir_ops.edge.aten._to_copy.default: 0,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
        })
        .run_method_and_compare_outputs()
        """

    def test_convert_with_nchw_after(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                y = x.to(torch.float)
                return self.conv(y)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .check_node_count(
                {
                    exir_ops.edge.aten._to_copy.default: 0,
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_convert_between_nchw(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                y = torch.nn.functional.interpolate(x, mode="bilinear", scale_factor=2)
                y = y.to(torch.float)
                return self.conv(y)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs()
        )

        """
        .to_edge()
        .check_node_count({
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 1,
        })
        .run_passes(self.PassStage)
        .check_node_count({
            exir_ops.edge.aten._to_copy.default: 0,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
        })
        .run_method_and_compare_outputs()
        """

    def test_fp16_convert_not_modified(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.to(torch.float16)
                y = y
                return y.to(torch.float)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 2,
                }
            )
            .run_passes(self.PassStage)
            .check_node_count(
                {
                    exir_ops.edge.aten._to_copy.default: 2,
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_dim_order_convert_not_modified(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.to(memory_format=torch.channels_last)

        inputs = (torch.randint(0, 255, (1, 3, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .to_edge()
            .check_node_count(
                {
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default: 1,
                }
            )
            .run_passes(self.PassStage)
            .check_node_count(
                {
                    exir_ops.edge.aten._to_copy.default: 1,
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                }
            )
            .run_method_and_compare_outputs()
        )
