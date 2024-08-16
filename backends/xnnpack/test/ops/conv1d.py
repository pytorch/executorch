# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.test_xnnpack_utils import randomize_bn

from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower
from executorch.exir.passes.constant_prop_pass import constant_prop_pass


class TestConv1d(unittest.TestCase):
    class Conv1d(torch.nn.Module):
        def __init__(self, dtype: torch.dtype = torch.float):
            groups = 1
            stride = (2,)
            padding = (1,)
            dilation = (1,)
            in_channels = 2
            out_channels = 1
            kernel_size = (3,)

            super().__init__()

            self.conv1d = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=True,
            ).to(dtype)

        def forward(self, x):
            return self.conv1d(x)

    class Conv1dBatchNormSequential(torch.nn.Module):
        def __init__(self):
            groups = 1
            stride = [1]
            padding = [1]
            dilation = [1]
            in_channels = 2
            out_channels = 2
            kernel_size = (3,)

            super().__init__()
            self.conv1 = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=True,
            )
            self.bn1 = randomize_bn(num_features=in_channels, dimensionality=1)
            self.conv2 = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=True,
            )
            self.bn2 = randomize_bn(num_features=in_channels, dimensionality=1)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            z = self.conv2(y)
            z = self.bn2(z)
            z = torch.add(y, z)
            return z

    def _test_conv1d(
        self,
        module,
        inputs,
        conv_count,
        quantized=False,
        dynamic_shape=None,
        passes=None,
        stage=None,
        skip_to_executorch=False,
    ):
        tester = (
            (
                Tester(module, inputs, dynamic_shape).quantize()
                if quantized
                else Tester(module, inputs)
            )
            .export()
            .run_passes(passes)
            .check_count({"torch.ops.aten.conv1d.default": conv_count})
            .to_edge_transform_and_lower(stage)
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        )
        # For some tests we want to skip to_executorch because otherwise it will require the
        # quantized operators to be loaded and we don't want to do that in the test.
        if not skip_to_executorch:
            tester.to_executorch().serialize().run_method_and_compare_outputs()

    def test_fp16_conv1d(self):
        inputs = (torch.randn(2, 2, 4).to(torch.float16),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(
            self.Conv1d(dtype=torch.float16),
            inputs,
            conv_count=1,
            dynamic_shape=dynamic_shapes,
        )

    def test_fp32_conv1d(self):
        inputs = (torch.randn(2, 2, 4),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(self.Conv1d(), inputs, 1, dynamic_shape=dynamic_shapes)

    def test_fp32_conv1d_batchnorm_seq(self):
        inputs = (torch.randn(2, 2, 4),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(
            self.Conv1dBatchNormSequential(), inputs, 2, dynamic_shape=dynamic_shapes
        )

    def test_qs8_conv1d(self):
        inputs = (torch.randn(2, 2, 4),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(
            self.Conv1d(), inputs, 1, quantized=True, dynamic_shape=dynamic_shapes
        )

    def test_qs8_conv1d_batchnorm_seq(self):
        inputs = (torch.randn(2, 2, 4),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(
            self.Conv1dBatchNormSequential(),
            inputs,
            2,
            quantized=True,
            dynamic_shape=dynamic_shapes,
        )

    def test_qs8_conv1d_with_floating_point_partitioner(self):
        inputs = (torch.randn(2, 2, 4),)
        dynamic_shapes = ({0: torch.export.Dim("batch", min=2, max=10)},)
        self._test_conv1d(
            self.Conv1d(),
            inputs,
            1,
            quantized=True,
            dynamic_shape=dynamic_shapes,
            stage=ToEdgeTransformAndLower(
                partitioners=[
                    XnnpackPartitioner(config_precisions=ConfigPrecisionType.FP32)
                ]
            ),
            passes=RunPasses(pass_functions=[constant_prop_pass]),
            skip_to_executorch=True,
        )
