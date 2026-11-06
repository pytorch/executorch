# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.config.generic_node_configs import (
    MeanDimConfig,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester, ToEdgeTransformAndLower
from executorch.export.utils import is_fbcode


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

    class GraphInputMeanDim(torch.nn.Module):
        def __init__(self, dims, keepdim=True):
            super().__init__()
            self.dims = dims
            self.keepdim = keepdim

        def forward(self, x):
            return torch.mean(x, self.dims, keepdim=self.keepdim)

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.rand(dim))

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    class ConvMeanDim(torch.nn.Module):
        def __init__(self, dims, keepdim=True):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.dims = dims
            self.keepdim = keepdim

        def forward(self, x):
            return torch.mean(self.conv(x), self.dims, keepdim=self.keepdim)

    def _test_mean_dim(self, inputs, dims=(-1, -2), keepdim=True, dynamic_shapes=None):
        (
            Tester(
                self.MeanDim(dims, keepdim=keepdim),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def _test_mean_dim_not_partitioned(self, inputs, dims, **kwargs):
        (
            Tester(self.MeanDim(dims, **kwargs), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def _test_qs8_mean_dim(self, inputs, dims):
        (
            Tester(self.MeanDim(dims), inputs)
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

    def test_fp16_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4).to(torch.float16),)
        self._test_mean_dim(inputs)

    def test_fp32_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs)

    def test_fp32_mean_dim_positive_dims(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs, dims=(2, 3))

    def test_fp32_mean_dim_single_innermost(self):
        inputs = (torch.randn(1, 16, 32, 32),)
        self._test_mean_dim(inputs, dims=(3,))

    def test_fp32_mean_dim_channels(self):
        inputs = (torch.randn(1, 16, 32, 32),)
        self._test_mean_dim(inputs, dims=(1,))

    def test_fp32_mean_dim_keepdim_false(self):
        inputs = (torch.randn(1, 16, 32, 32),)
        self._test_mean_dim(inputs, keepdim=False)

    def test_fp32_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_mean_dim(inputs, dims=(-1,))

    def test_fp16_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128).to(torch.float16),)
        self._test_mean_dim(inputs, dims=(-1,))

    def test_fp16_mean_dim_middle_dim_3d(self):
        inputs = (torch.randn(2, 64, 128).to(torch.float16),)
        self._test_mean_dim(inputs, dims=(1,))

    def test_fp32_mean_dim_3d_keepdim_false(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_mean_dim(inputs, dims=(-1, -2), keepdim=False)

    def test_fp32_mean_dim_1d(self):
        inputs = (torch.randn(64),)
        for keepdim in (True, False):
            with self.subTest(keepdim=keepdim):
                self._test_mean_dim(inputs, dims=(0,), keepdim=keepdim)

    def test_fp32_mean_dim_2d(self):
        inputs = (torch.randn(128, 128),)
        self._test_mean_dim(inputs, dims=(0,))

    def test_fp32_mean_dim_5d(self):
        inputs = (torch.randn(2, 4, 8, 16, 16),)
        self._test_mean_dim(inputs, dims=(1, 3))

    def test_fp32_mean_dim_all_dims(self):
        inputs = (torch.randn(2, 7, 16),)
        for dims in (None, []):
            for keepdim in (True, False):
                with self.subTest(dims=dims, keepdim=keepdim):
                    self._test_mean_dim(inputs, dims=dims, keepdim=keepdim)

    def test_fp32_mean_dim_small_single_output(self):
        inputs = (torch.randn(1, 1, 64),)
        self._test_mean_dim(inputs, dims=(-1,))

    def test_fp32_mean_dim_small_multi_output_not_partitioned(self):
        """
        A static mean with more than one output and fewer than
        MeanDimConfig.min_multi_row_input_numel input elements stays on the
        portable kernel, which can be faster than waking XNNPACK's threadpool.
        """
        inputs = (torch.randn(2, 7, 16),)
        self._test_mean_dim_not_partitioned(inputs, dims=(-1,))

    def test_fp32_mean_dim_min_multi_row_input_numel(self):
        rows = 8
        cols = MeanDimConfig.min_multi_row_input_numel // rows
        self._test_mean_dim((torch.randn(1, rows, cols),), dims=(-1,))
        self._test_mean_dim_not_partitioned(
            (torch.randn(1, rows, cols - 1),), dims=(-1,)
        )

    def test_fp32_mean_dim_last_dim_3d_dynamic(self):
        inputs = (torch.randn(2, 7, 16),)
        seq = torch.export.Dim("seq", min=1, max=32)
        self._test_mean_dim(inputs, dims=(-1,), dynamic_shapes=({1: seq},))

    def test_fp32_mean_dim_dynamic_reduced_dim(self):
        inputs = (torch.randn(2, 7, 16),)
        seq = torch.export.Dim("seq", min=1, max=32)
        self._test_mean_dim(inputs, dims=(1,), dynamic_shapes=({1: seq},))

    def test_fp32_rms_norm(self):
        inputs = (torch.randn(2, 64, 128),)
        (
            Tester(self.RMSNorm(128), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_conv_mean_dim(self):
        """
        The convolution output is NHWC inside the delegate, and a mean that is
        not Global Average Pooling reads it back in NCHW.
        """
        inputs = (torch.randn(1, 3, 34, 34),)
        for dims, keepdim in (((1,), True), ((-1, -2), False)):
            with self.subTest(dims=dims, keepdim=keepdim):
                (
                    Tester(self.ConvMeanDim(dims, keepdim), inputs)
                    .export()
                    .to_edge_transform_and_lower()
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def _export_and_lower(self, module, inputs, dynamic_shapes=None, enable_bf16=True):
        return (
            Tester(module, inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    partitioners=[XnnpackPartitioner(enable_bf16=enable_bf16)]
                )
            )
        )

    def _test_bf16_mean_dim(self, module, inputs, dynamic_shapes=None, num_runs=1):
        (
            self._export_and_lower(module.to(torch.bfloat16), inputs, dynamic_shapes)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=num_runs, atol=0.01, rtol=0.01)
        )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.MeanDim((-1,)), inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_middle_dim_keepdim_false(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.MeanDim((1,), keepdim=False), inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_global_avg_pool_shape(self):
        inputs = (torch.randn(1, 16, 32, 32).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.MeanDim((-1, -2)), inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_single_output(self):
        inputs = (torch.randn(1, 1, 64).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.MeanDim((-1,)), inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_of_graph_input(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.GraphInputMeanDim((-1,)), inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_last_dim_3d_dynamic(self):
        inputs = (torch.randn(2, 7, 16).to(torch.bfloat16),)
        seq = torch.export.Dim("seq", min=1, max=32)
        self._test_bf16_mean_dim(
            self.MeanDim((-1,)), inputs, dynamic_shapes=({1: seq},), num_runs=5
        )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_rms_norm(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_bf16_mean_dim(self.RMSNorm(128), inputs)

    def test_bf16_mean_dim_not_partitioned(self):
        """
        A small static bf16 mean with more than one output stays on the portable
        kernel, including a Global Average Pooling shaped one, and bf16 needs
        enable_bf16.
        """
        cases = (
            ("small_multi_output", (-1,), (2, 7, 16), True),
            ("small_global_avg_pool_shape", (-1, -2), (1, 5, 4, 4), True),
            ("enable_bf16_false", (-1,), (2, 64, 128), False),
        )
        for name, dims, shape, enable_bf16 in cases:
            with self.subTest(name):
                inputs = (torch.randn(*shape).to(torch.bfloat16),)
                self._export_and_lower(
                    self.MeanDim(dims), inputs, enable_bf16=enable_bf16
                ).check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})

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
        self._test_qs8_mean_dim(inputs, dims=(-1, -2))

    def test_qs8_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_qs8_mean_dim(inputs, dims=(-1,))

    def test_qs8_mean_dim_middle_dim_3d(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_qs8_mean_dim(inputs, dims=(1,))
