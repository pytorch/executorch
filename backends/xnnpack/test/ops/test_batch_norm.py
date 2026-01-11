# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import randomize_bn
from executorch.backends.xnnpack.test.tester import Tester


class TestBatchNorm(unittest.TestCase):
    """
    End-to-end tests for standalone BatchNorm operators lowered to XNNPACK.
    """

    def setUp(self):
        torch._dynamo.reset()

    class BatchNorm1dNC(torch.nn.Module):
        """BatchNorm1d with NC input (batch, channels)."""

        def __init__(self, num_features: int):
            super().__init__()
            self.num_features = num_features
            self.bn = torch.nn.BatchNorm1d(num_features)

        def forward(self, x):
            return self.bn(x)

        def get_inputs(self):
            return (torch.randn(2, self.num_features),)

    class BatchNorm1dNCL(torch.nn.Module):
        """BatchNorm1d with NCL input (batch, channels, length)."""

        def __init__(self, num_features: int):
            super().__init__()
            self.num_features = num_features
            self.bn = torch.nn.BatchNorm1d(num_features)

        def forward(self, x):
            return self.bn(x)

        def get_inputs(self):
            return (torch.randn(2, self.num_features, 8),)

    class BatchNorm2d(torch.nn.Module):
        """BatchNorm2d with NCHW input (batch, channels, height, width)."""

        def __init__(self, num_features: int, dtype: torch.dtype = torch.float):
            super().__init__()
            self.num_features = num_features
            self.dtype = dtype
            self.bn = torch.nn.BatchNorm2d(num_features).to(dtype)

        def forward(self, x):
            return self.bn(x)

        def get_inputs(self):
            return (torch.randn(2, self.num_features, 4, 4).to(self.dtype),)

    def _test_batch_norm(self, model: torch.nn.Module):
        """
        Test that a standalone BatchNorm is lowered to XNNPACK via decomposition
        to depthwise convolution.
        """
        # Warm up batch norm running stats
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(*model.get_inputs())

        (
            Tester(model, model.get_inputs())
            .export()
            .to_edge_transform_and_lower()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class LinearReluBatchNorm(torch.nn.Module):
        """
        Linear followed by ReLU, BatchNorm, residual add, and a second Linear.
        The BatchNorm is standalone (not fused) because ReLU breaks the fusion pattern.
        """

        def __init__(self, features: int):
            super().__init__()
            self.features = features
            self.linear1 = torch.nn.Linear(features, features)
            self.relu = torch.nn.ReLU()
            self.bn = randomize_bn(features, dimensionality=1)
            self.linear2 = torch.nn.Linear(features, features)

        def forward(self, x):
            y = self.linear1(x)
            y = self.relu(y)
            y = self.bn(y)
            y = y + x
            y = self.linear2(y)
            return y

        def get_inputs(self):
            return (torch.randn(2, self.features),)

    def test_fp32_linear_relu_batch_norm(self):
        """
        Test Linear + ReLU + BatchNorm where the BatchNorm is standalone (not fused
        with linear) because ReLU breaks the fusion pattern. The standalone BatchNorm
        should be decomposed to depthwise convolution.
        """
        model = self.LinearReluBatchNorm(features=8)
        model.eval()

        (
            Tester(model, model.get_inputs())
            .export()
            .to_edge_transform_and_lower()
            # BatchNorm should be decomposed (not present in the graph)
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_batch_norm_nc(self):
        """Test BatchNorm1d with NC input is lowered to XNNPACK."""
        self._test_batch_norm(self.BatchNorm1dNC(num_features=3))

    def test_fp32_batch_norm_ncl(self):
        """Test BatchNorm1d with NCL input is lowered to XNNPACK."""
        self._test_batch_norm(self.BatchNorm1dNCL(num_features=3))

    def test_fp32_batch_norm_nchw(self):
        """Test BatchNorm2d with NCHW input is lowered to XNNPACK."""
        self._test_batch_norm(self.BatchNorm2d(num_features=3))

    def test_fp16_batch_norm_nchw(self):
        """Test BatchNorm2d with fp16 NCHW input is lowered to XNNPACK."""
        self._test_batch_norm(self.BatchNorm2d(num_features=3, dtype=torch.float16))

    class BatchNorm3d(torch.nn.Module):
        """BatchNorm3d with NCDHW input (batch, channels, depth, height, width)."""

        def __init__(self, num_features: int):
            super().__init__()
            self.num_features = num_features
            self.bn = torch.nn.BatchNorm3d(num_features)

        def forward(self, x):
            return self.bn(x)

        def get_inputs(self):
            return (torch.randn(2, self.num_features, 4, 4, 4),)

    def test_fp32_batch_norm3d_not_partitioned(self):
        """Test that BatchNorm3d is NOT partitioned to XNNPACK (unsupported)."""
        model = self.BatchNorm3d(num_features=3)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(*model.get_inputs())

        (
            Tester(model, model.get_inputs())
            .export()
            .to_edge_transform_and_lower()
            # BatchNorm3d should remain in the graph (not lowered to XNNPACK)
            .check(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            # No delegate call should be present since nothing was partitioned
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class Conv2dReluBatchNorm(torch.nn.Module):
        """Conv2d followed by ReLU and then BatchNorm (standalone BN, not fused)."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.in_channels = in_channels
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.bn = randomize_bn(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.bn(x)
            return x

        def get_inputs(self):
            return (torch.randn(2, self.in_channels, 8, 8),)

    def test_fp32_conv2d_relu_batch_norm(self):
        """
        Test Conv2d + ReLU + BatchNorm where the BatchNorm is standalone (not fused
        with conv) because ReLU breaks the fusion pattern. The standalone BatchNorm
        should be decomposed to depthwise convolution.
        """
        model = self.Conv2dReluBatchNorm(in_channels=3, out_channels=8)
        model.eval()

        (
            Tester(model, model.get_inputs())
            .export()
            .to_edge_transform_and_lower()
            # BatchNorm should be decomposed (not present in the graph)
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class Conv2dBatchNorm(torch.nn.Module):
        """Conv2d followed by BatchNorm (fuseable pattern)."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.in_channels = in_channels
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = randomize_bn(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return x

        def get_inputs(self):
            return (torch.randn(2, self.in_channels, 8, 8),)

    def test_fp32_conv2d_batch_norm_fused(self):
        """
        Test Conv2d + BatchNorm where the BatchNorm is fused into the Conv2d.
        This tests the existing fusion path (not decomposition).
        """
        model = self.Conv2dBatchNorm(in_channels=3, out_channels=8)
        model.eval()

        (
            Tester(model, model.get_inputs())
            .export()
            .to_edge_transform_and_lower()
            # BatchNorm should be fused into conv (not present in the graph)
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
