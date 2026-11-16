# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
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

    class ConvSubtractMean(torch.nn.Module):
        """The centering in ConvNeXt's LayerNorm2d, after a convolution."""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            y = self.conv(x)
            return y - y.mean(1, keepdim=True)

    class ConvMeanConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3)
            self.conv2 = torch.nn.Conv2d(1, 8, 3)

        def forward(self, x):
            return self.conv2(self.conv1(x).mean(1, keepdim=True))

    @staticmethod
    def _xnnpack_mean_nodes(global_avg_pool):
        """
        For a mean of a contiguous graph input, Global Average Pooling is the
        only lowering of a 4D mean over [2, 3] with keepdim, with its input
        transposed to NHWC and its output back, as on main; every other mean is
        one XNNStaticMean with no transpose.
        """
        if global_avg_pool:
            return {
                "XNNGlobalAvgPooling2d": 1,
                "XNNStaticMean": 0,
                "XNNStaticTranspose": 2,
            }
        return {
            "XNNGlobalAvgPooling2d": 0,
            "XNNStaticMean": 1,
            "XNNStaticTranspose": 0,
        }

    def _test_mean_dim(
        self,
        inputs,
        dims=(-1, -2),
        keepdim=True,
        dynamic_shapes=None,
        global_avg_pool=False,
        num_runs=1,
        module=None,
        xnnpack_nodes=None,
    ):
        """
        A bf16 mean serializes as a bf16 XNNStaticMean with no converts; the
        runtime computes it in fp32.
        """
        bf16 = inputs[0].dtype == torch.bfloat16
        module = module or self.MeanDim(dims, keepdim=keepdim)
        if xnnpack_nodes is None:
            xnnpack_nodes = self._xnnpack_mean_nodes(global_avg_pool)
        if bf16:
            module = module.to(torch.bfloat16)
            xnnpack_nodes = {"XNNConvert": 0, **xnnpack_nodes}
        tol = 1e-2 if bf16 else 1e-3
        (
            Tester(module, inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    partitioners=[XnnpackPartitioner(enable_bf16=bf16)]
                )
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .check_xnnpack_node_count(xnnpack_nodes)
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=num_runs, atol=tol, rtol=tol)
        )

    def _test_qs8_mean_dim(
        self, inputs, dims, global_avg_pool=False, xnnpack_nodes=None
    ):
        """The input quantize and the output dequantize are the two converts."""
        if xnnpack_nodes is None:
            xnnpack_nodes = self._xnnpack_mean_nodes(global_avg_pool)
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
            .check_xnnpack_node_count({**xnnpack_nodes, "XNNConvert": 2})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=1)
        )

    def test_fp16_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4).to(torch.float16),)
        self._test_mean_dim(inputs, global_avg_pool=True)

    def test_fp32_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs, global_avg_pool=True)

    def test_fp32_mean_dim_positive_dims(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_mean_dim(inputs, dims=(2, 3), global_avg_pool=True)

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

    def test_fp32_mean_dim_6d(self):
        inputs = (torch.randn(2, 2, 4, 4, 8, 16),)
        self._test_mean_dim(inputs, dims=(1, 4))

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

    def test_fp32_mean_dim_small_multi_output(self):
        inputs = (torch.randn(2, 7, 16),)
        self._test_mean_dim(inputs, dims=(-1,))

    def test_fp32_mean_dim_last_dim_3d_dynamic(self):
        inputs = (torch.randn(2, 7, 16),)
        seq = torch.export.Dim("seq", min=1, max=32)
        self._test_mean_dim(inputs, dims=(-1,), dynamic_shapes=({1: seq},), num_runs=5)

    def test_fp32_mean_dim_dynamic_reduced_dim(self):
        inputs = (torch.randn(2, 7, 16),)
        seq = torch.export.Dim("seq", min=1, max=32)
        self._test_mean_dim(inputs, dims=(1,), dynamic_shapes=({1: seq},), num_runs=5)

    def test_fp32_rms_norm(self):
        inputs = (torch.randn(2, 64, 128),)
        (
            Tester(self.RMSNorm(128), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mean_dim",
                    "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar",
                    "executorch_exir_dialects_edge__ops_aten_rsqrt_default",
                    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                ]
            )
            .check_xnnpack_node_count({"XNNStaticMean": 1, "XNNGlobalAvgPooling2d": 0})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_conv_mean_dim(self):
        """
        The convolution output is NHWC inside the delegate. Global Average
        Pooling reads it as is, and any other mean reads it back in NCHW.
        """
        inputs = (torch.randn(1, 3, 34, 34),)
        cases = (
            (
                (-1, -2),
                True,
                {
                    "XNNGlobalAvgPooling2d": 1,
                    "XNNStaticMean": 0,
                    "XNNStaticTranspose": 2,
                },
            ),
            ((1,), True, {"XNNStaticMean": 1, "XNNStaticTranspose": 2}),
            ((-1, -2), False, {"XNNStaticMean": 1, "XNNStaticTranspose": 2}),
        )
        for dims, keepdim, xnnpack_nodes in cases:
            with self.subTest(dims=dims, keepdim=keepdim):
                (
                    Tester(self.ConvMeanDim(dims, keepdim), inputs)
                    .export()
                    .to_edge_transform_and_lower()
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
                    .check_xnnpack_node_count(xnnpack_nodes)
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )

    def test_qs8_conv_mean_dim(self):
        inputs = (torch.randn(1, 3, 34, 34),)
        for dims, keepdim in (((1,), True), ((-1, -2), False)):
            with self.subTest(dims=dims, keepdim=keepdim):
                (
                    Tester(self.ConvMeanDim(dims, keepdim).eval(), inputs)
                    .quantize()
                    .export()
                    .to_edge_transform_and_lower()
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .check_not(
                        [
                            "executorch_exir_dialects_edge__ops_aten_mean_dim",
                            "torch.ops.quantized_decomposed",
                        ]
                    )
                    .check_xnnpack_node_count(
                        {"XNNStaticMean": 1, "XNNStaticTranspose": 2, "XNNConvert": 2}
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs(qtol=1)
                )

    def test_conv_mean_meets_nhwc(self):
        """
        A mean over channels after a convolution reads the NHWC convolution
        output back in NCHW, and its result goes back to NHWC to meet the
        convolution output or feed another convolution. With the input, the
        mean's input and output, and the delegate output, that is 4 transposes;
        main has 2 for ConvSubtractMean (3 in qs8), split over two delegates
        around a portable mean, and 4 for ConvMeanConv. In qs8, the mean can
        round one step (about 0.016 here) away from the reference, and that
        difference reaches the output.
        """
        inputs = (torch.randn(1, 3, 18, 18),)
        for module in (self.ConvSubtractMean(), self.ConvMeanConv()):
            for quantize in (False, True):
                with self.subTest(type(module).__name__, quantize=quantize):
                    tester = Tester(module.eval(), inputs)
                    if quantize:
                        tester.quantize()
                    (
                        tester.export()
                        .to_edge_transform_and_lower()
                        .check_count(
                            {"torch.ops.higher_order.executorch_call_delegate": 1}
                        )
                        .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
                        .check_xnnpack_node_count(
                            {"XNNStaticMean": 1, "XNNStaticTranspose": 4}
                        )
                        .to_executorch()
                        .serialize()
                        .run_method_and_compare_outputs(atol=2e-2 if quantize else 1e-3)
                    )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_mean_dim(inputs, dims=(-1,))

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_middle_dim_keepdim_false(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_mean_dim(inputs, dims=(1,), keepdim=False)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_global_avg_pool_shape(self):
        """XNNPACK has no bf16 Global Average Pooling."""
        for shape in ((1, 16, 32, 32), (1, 5, 4, 4)):
            with self.subTest(shape=shape):
                inputs = (torch.randn(*shape).to(torch.bfloat16),)
                self._test_mean_dim(inputs)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_channels_last_input(self):
        """A channels_last input is copied to NCHW for the static mean."""
        inputs = (
            torch.randn(1, 16, 8, 8)
            .to(torch.bfloat16)
            .to(memory_format=torch.channels_last),
        )
        self._test_mean_dim(
            inputs,
            module=self.GraphInputMeanDim((-1, -2)),
            xnnpack_nodes={
                "XNNGlobalAvgPooling2d": 0,
                "XNNStaticMean": 1,
                "XNNStaticTranspose": 1,
            },
        )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_small(self):
        for shape in ((1, 1, 64), (2, 7, 16)):
            with self.subTest(shape=shape):
                inputs = (torch.randn(*shape).to(torch.bfloat16),)
                self._test_mean_dim(inputs, dims=(-1,))

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_all_dims(self):
        inputs = (torch.randn(2, 7, 16).to(torch.bfloat16),)
        for keepdim in (True, False):
            with self.subTest(keepdim=keepdim):
                self._test_mean_dim(inputs, dims=None, keepdim=keepdim)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_of_graph_input(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        self._test_mean_dim(inputs, module=self.GraphInputMeanDim((-1,)))

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_mean_dim_dynamic(self):
        seq = torch.export.Dim("seq", min=1, max=32)
        for dims in ((-1,), (1,)):
            with self.subTest(dims=dims):
                inputs = (torch.randn(2, 7, 16).to(torch.bfloat16),)
                self._test_mean_dim(
                    inputs, dims=dims, dynamic_shapes=({1: seq},), num_runs=5
                )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_rms_norm(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        # The one convert casts the eps constant to bf16.
        self._test_mean_dim(
            inputs,
            module=self.RMSNorm(128),
            xnnpack_nodes={"XNNStaticMean": 1, "XNNConvert": 1},
        )

    def test_bf16_mean_dim_not_partitioned_without_enable_bf16(self):
        inputs = (torch.randn(2, 64, 128).to(torch.bfloat16),)
        (
            Tester(self.MeanDim((-1,)).to(torch.bfloat16), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_fp16_mean_dim_dtype_argument(self):
        """
        An fp16 mean with an fp32 dtype passes the generic dtype checks, so
        MeanDimConfig's own dtype check keeps it on the portable kernel.
        """
        inputs = (torch.randn(2, 7, 16).to(torch.float16),)
        (
            Tester(self.MeanDim((-1,), dtype=torch.float32), inputs)
            .export()
            .check_count({"torch.ops.aten.mean.dim": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mean_dim": 1})
        )

    def test_qs8_mean_dim(self):
        inputs = (torch.randn(1, 5, 4, 4),)
        self._test_qs8_mean_dim(inputs, dims=(-1, -2), global_avg_pool=True)

    def test_qs8_mean_dim_last_dim_3d(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_qs8_mean_dim(inputs, dims=(-1,))

    def test_qs8_mean_dim_channels_last_input(self):
        """
        The quantized channels_last input is NHWC, so a mean over NCHW dims
        reads it through a transpose, and Global Average Pooling reads it as is
        and only transposes its output back.
        """
        inputs = (torch.randn(1, 5, 4, 4).to(memory_format=torch.channels_last),)
        for dims, xnnpack_nodes in (
            ((1,), {"XNNStaticMean": 1, "XNNStaticTranspose": 1}),
            ((-1, -2), {"XNNGlobalAvgPooling2d": 1, "XNNStaticTranspose": 1}),
        ):
            with self.subTest(dims=dims):
                self._test_qs8_mean_dim(inputs, dims=dims, xnnpack_nodes=xnnpack_nodes)

    def test_qs8_mean_dim_middle_dim_3d(self):
        inputs = (torch.randn(2, 64, 128),)
        self._test_qs8_mean_dim(inputs, dims=(1,))
