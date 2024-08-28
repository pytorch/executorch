# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMeanDim(unittest.TestCase):
    class MeanDim(torch.nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            y = x + x
            z = torch.mean(y, self.dims, keepdim=True)
            return z

    def _test_mean_dim(self, inputs):
        (
            Tester(self.MeanDim((-1, -2)), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4).to(torch.float16),)
        self._test_mean_dim(inputs)

    def test_fp32_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs)

    def test_fp32_mean_dim_unsupported(self):
        """
        XNNPack mean.dim implementation only supports innermost two dimensions. As such,
        we expect it to fail to partition when dim=(3).
        """
        inputs = (torch.randn(1, 5, 4, 4),)
        (
            Tester(self.MeanDim((3)), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_fp32_mean_dim_unsupported_3d(self):
        """
        XNNPack mean.dim implementation only supports 4D tensors.
        """
        inputs = (torch.randn(1, 5, 4),)
        (
            Tester(self.MeanDim((-1, -2)), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_qs8_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        (
            Tester(self.MeanDim((-1, -2)), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    torch.ops.aten.mean.dim: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mean_dim",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=1)
        )
