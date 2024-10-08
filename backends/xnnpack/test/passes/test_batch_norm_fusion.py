# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.xnnpack._passes.fuse_batch_norm_with_conv import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestBatchNormFusion(unittest.TestCase):
    PassStage = RunPasses([FuseBatchNormWithConvPass])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"

    class ModelConvBN(torch.nn.Module):
        def __init__(
            self, in_features: int, out_features: int, kernel_size: Tuple[int, int]
        ):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)
            self.bn = torch.nn.BatchNorm2d(out_features)

        def forward(self, x):
            y = self.conv2d(x)
            y = self.bn(y)
            y = self.conv2d(y)
            y = y + y
            return self.bn(y)

    def test_fp32_batch_norm_fusion(self):
        (
            Tester(self.ModelConvBN(2, 2, (2, 2)).eval(), (torch.randn(2, 2, 4, 4),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 1})
            .run_method_and_compare_outputs()
        )

    def test_q8_batch_norm_fusion(self):
        (
            Tester(self.ModelConvBN(2, 2, (2, 2)).eval(), (torch.randn(2, 2, 4, 4),))
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 1})
            .run_method_and_compare_outputs()
        )

    def test_fp32_batch_norm_no_fusion_doesnt_partition(self):
        """
        We do not currently support standalone batch norms (i.e. batch norms that are
        not fused with a conv). This is planned, but until implemented, this test ensures
        that we do not partition the standalone batch norm and then fail to lower.
        """

        class BN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                return self.bn(x)

        (
            Tester(BN(), (torch.randn(2, 2, 4, 4),))
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .partition()
            .check_count({self.bn_name: 1})
        )
