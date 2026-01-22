# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.xnnpack._passes.fuse_batch_norm import FuseBatchNormPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestBatchNormFusion(unittest.TestCase):
    PassStage = RunPasses([FuseBatchNormPass])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"

    def setUp(self):
        torch._dynamo.reset()

    class ModelConvBN(torch.nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: Tuple[int, int],
            transpose: bool,
        ):
            super().__init__()
            op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
            self.conv2d = op(in_features, out_features, kernel_size)
            self.bn = torch.nn.BatchNorm2d(out_features)
            self.forward(torch.randn(2, 2, 4, 4) * 2 + 2)  # update the BN stats

        def forward(self, x):
            y = self.conv2d(x)
            y = self.bn(y)
            y = self.conv2d(y)
            y = y + y
            return self.bn(y)

    class ModelLinearBN(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            op = torch.nn.Linear
            self.linear = op(in_features, out_features, bias=bias)
            self.bn = torch.nn.BatchNorm1d(out_features)
            self.forward(torch.randn(2, 2) * 2 + 2)  # update the BN stats

        def forward(self, x):
            y = self.linear(x)
            y = self.bn(y)
            y = self.linear(y)
            y = y + y
            return self.bn(y)

    class ModelConv3dBN(torch.nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: Tuple[int, int, int],
        ):
            super().__init__()
            op = torch.nn.Conv3d
            self.conv3d = op(in_features, out_features, kernel_size)
            self.bn = torch.nn.BatchNorm3d(out_features)
            self.forward(torch.randn(2, 2, 4, 4, 4) * 2 + 2)  # update the BN stats

        def forward(self, x):
            y = self.conv3d(x)
            y = self.bn(y)
            y = self.conv3d(y)
            y = y + y
            return self.bn(y)

    def test_fp32_conv_batch_norm_fusion(self):
        for transpose in [False, True]:
            (
                Tester(
                    self.ModelConvBN(2, 2, (2, 2), transpose).eval(),
                    (torch.randn(2, 2, 4, 4),),
                )
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check_count({self.bn_name: 1})
                .run_method_and_compare_outputs()
            )

    def test_q8_conv_batch_norm_fusion(self):
        for transpose in [False, True]:
            (
                Tester(
                    self.ModelConvBN(2, 2, (2, 2), transpose).eval(),
                    (torch.randn(2, 2, 4, 4),),
                )
                .quantize()
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check_count({self.bn_name: 1})
                .run_method_and_compare_outputs()
            )

    def test_fp32_linear_batch_norm_fusion(self):
        for bias in [True, False]:
            (
                Tester(
                    self.ModelLinearBN(2, 2, bias).eval(),
                    (torch.randn(2, 2),),
                )
                .export()
                .to_edge_transform_and_lower()
                .check_count({self.bn_name: 0})
                .run_method_and_compare_outputs()
            )

    def test_fp32_conv3d_batch_norm_doesnt_partition(self):
        """
        Conv3d is not currently supported by XNNPACK. We also don't support standalone
        batch norms yet (i.e. batch norms that are not fused with a conv). As such, we don't
        want to partition the standalone batch norm and then fail to lower.
        """
        (
            Tester(self.ModelConv3dBN(2, 2, (2, 2, 2)), (torch.randn(2, 2, 4, 4, 4),))
            .export()
            .dump_artifact()
            .to_edge_transform_and_lower()
            .check_count({self.bn_name: 2})
            .run_method_and_compare_outputs()
        )
