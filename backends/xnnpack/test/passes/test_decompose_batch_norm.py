# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.decompose_batch_norm import DecomposeBatchNorm
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestDecomposeBatchNorm(unittest.TestCase):
    PassStage = RunPasses([DecomposeBatchNorm])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
    conv_name = "executorch_exir_dialects_edge__ops_aten_convolution_default"

    def setUp(self):
        torch._dynamo.reset()

    class BatchNorm1dNC(torch.nn.Module):
        """Simple BatchNorm1d module with NC input (no spatial dimension)."""

        def __init__(self, num_features: int):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(num_features)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    class BatchNorm1dNCL(torch.nn.Module):
        """Simple BatchNorm1d module with NCL input."""

        def __init__(self, num_features: int):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(num_features)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features, 4) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    class BatchNorm2d(torch.nn.Module):
        """Simple BatchNorm2d module with NCHW input."""

        def __init__(self, num_features: int):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(num_features)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features, 4, 4) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    def test_fp32_batch_norm_nc(self):
        """Test that BatchNorm1d with NC input is decomposed to convolution."""
        (
            Tester(
                self.BatchNorm1dNC(3).eval(),
                (torch.randn(2, 3),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )

    def test_fp32_batch_norm_ncl(self):
        """Test that BatchNorm1d with NCL input is decomposed to convolution."""
        (
            Tester(
                self.BatchNorm1dNCL(3).eval(),
                (torch.randn(2, 3, 4),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )

    def test_fp32_batch_norm_nchw(self):
        """Test that BatchNorm2d with NCHW input is decomposed to convolution."""
        (
            Tester(
                self.BatchNorm2d(3).eval(),
                (torch.randn(2, 3, 4, 4),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
