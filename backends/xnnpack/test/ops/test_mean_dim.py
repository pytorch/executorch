# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMeanDim(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class MeanDim(torch.nn.Module):
        def __init__(self, dims, keepdim=True, dtype=None):
            super().__init__()
            self.dims = dims
            self.keepdim = keepdim
            self.dtype = dtype

        def forward(self, x):
            y = x + x
            if self.dtype is None:
                z = torch.mean(y, self.dims, keepdim=self.keepdim)
            else:
                z = torch.mean(y, self.dims, keepdim=self.keepdim, dtype=self.dtype)
            return z

    def _test_mean_dim(self, inputs, dims=(-1, -2)):
        (
            Tester(self.MeanDim(dims), inputs)
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

    def test_fp32_mean_dim_positive_dims(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs, dims=(2, 3))

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

    def test_fp32_mean_dim_unsupported_keepdim_false(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        (
            Tester(self.MeanDim((-1, -2), keepdim=False), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_fp32_mean_dim_unsupported_dtype(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        (
            Tester(self.MeanDim((-1, -2), dtype=torch.float64), inputs)
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
