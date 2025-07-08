# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.convert_batch_norm_to_depthwise_conv import (
    ConvertBatchNormToDepthwiseConvPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestBatchNormToDepthwiseConv(unittest.TestCase):
    PassStage = RunPasses([ConvertBatchNormToDepthwiseConvPass])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
    conv_name = "executorch_exir_dialects_edge__ops_aten_convolution_default"

    def setUp(self):
        torch._dynamo.reset()

    def test_standalone_batch_norm_conversion(self):
        """Test that standalone batch norm is converted to depthwise convolution."""

        class StandaloneBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(4)
                # Initialize batch norm with some data to set proper statistics
                with torch.no_grad():
                    dummy_input = torch.randn(1, 4, 8, 8)
                    self.forward(dummy_input)

            def forward(self, x):
                return self.bn(x)

        (
            Tester(StandaloneBN().eval(), (torch.randn(1, 4, 8, 8),))
            .export()
            .to_edge()
            .check_count({self.bn_name: 1, self.conv_name: 0})
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 0, self.conv_name: 1})  # BN converted to conv
            .run_method_and_compare_outputs()
        )

    def test_batch_norm_after_conv_not_converted(self):
        """Test that batch norm after conv is not converted (should be handled by fusion)."""

        class ConvBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(4)
                # Initialize with dummy data
                with torch.no_grad():
                    dummy_input = torch.randn(1, 4, 8, 8)
                    self.forward(dummy_input)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        (
            Tester(ConvBN().eval(), (torch.randn(1, 4, 8, 8),))
            .export()
            .to_edge()
            .check_count({self.bn_name: 1, self.conv_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 1, self.conv_name: 1})  # No change - fusion should handle this
            .run_method_and_compare_outputs()
        )

    def test_multiple_standalone_batch_norms(self):
        """Test multiple standalone batch norms in sequence."""

        class MultipleBN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(4)
                self.bn2 = torch.nn.BatchNorm2d(4)
                # Initialize with dummy data
                with torch.no_grad():
                    dummy_input = torch.randn(1, 4, 8, 8)
                    self.forward(dummy_input)

            def forward(self, x):
                x = self.bn1(x)
                x = torch.relu(x)
                x = self.bn2(x)
                return x

        (
            Tester(MultipleBN().eval(), (torch.randn(1, 4, 8, 8),))
            .export()
            .to_edge()
            .check_count({self.bn_name: 2, self.conv_name: 0})
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 0, self.conv_name: 2})  # Both BNs converted to conv
            .run_method_and_compare_outputs()
        )


if __name__ == "__main__":
    unittest.main()
