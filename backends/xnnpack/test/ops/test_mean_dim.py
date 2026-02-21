# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
        def __init__(self, dims, keepdim=True):
            super().__init__()
            self.dims = dims
            self.keepdim = keepdim

        def forward(self, x):
            y = x + x
            z = torch.mean(y, self.dims, keepdim=self.keepdim)
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

    def _test_mean_dim_single(self, inputs):
        (
            Tester(self.MeanDim(-1), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_mean_dim_single(self):
        """
        Test that mean.dim with dim=[-1] and keepdim=True on 3D tensors is supported
        after the MeanDimRewritePass transforms it to the 4D case.
        """
        inputs = (torch.randn(1, 5, 4).to(torch.float16),)
        self._test_mean_dim_single(inputs)

    def test_fp32_mean_dim_single(self):
        """
        Test that mean.dim with dim=[-1] and keepdim=True on 3D tensors is supported
        after the MeanDimRewritePass transforms it to the 4D case.
        """
        inputs = (torch.randn(1, 5, 4),)
        self._test_mean_dim_single(inputs)

    def test_fp32_mean_dim_single_unsupported_without_keepdim(self):
        """
        Test that mean.dim with dim=[-1] but keepdim=False is not supported.
        """

        class MeanDimSingleNoKeepDim(torch.nn.Module):
            def forward(self, x):
                y = x + x
                z = torch.mean(y, -1, keepdim=False)
                return z

        inputs = (torch.randn(1, 5, 4),)
        (
            Tester(MeanDimSingleNoKeepDim(), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_fp32_mean_dim_single_unsupported_4d(self):
        """
        Test that mean.dim with dim=[-1] on 4D tensors is not supported
        (only 3D tensors are supported for single dim mean).
        """
        inputs = (torch.randn(1, 5, 4, 4),)
        (
            Tester(self.MeanDim(-1), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_qs8_mean_dim_single(self):
        """
        Test that quantized mean.dim with dim=[-1] and keepdim=True on 3D tensors is supported
        after the MeanDimRewritePass transforms it to the 4D case.
        """
        inputs = (torch.randn(1, 5, 4),)
        (
            Tester(self.MeanDim(-1), inputs)
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

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            """
            From https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L34
            """
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight

    def test_fp32_rmsnorm(self):
        """
        Test RMSNorm which uses mean.dim with dim=[-1] and keepdim=True.
        This validates that the MeanDimRewritePass correctly handles RMSNorm normalization.
        """
        dim = 128
        inputs = (torch.randn(1, 10, dim),)
        (
            Tester(self.RMSNorm(dim), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
