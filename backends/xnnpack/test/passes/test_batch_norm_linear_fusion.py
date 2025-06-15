# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.fuse_batch_norm_with_linear import (
    FuseBatchNormWithLinearPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestBatchNormLinearFusion(unittest.TestCase):
    PassStage = RunPasses([FuseBatchNormWithLinearPass])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"

    def setUp(self):
        torch._dynamo.reset()

    class ModelLinearBN(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            op = torch.nn.Linear
            self.linear = op(in_features, out_features)
            self.bn = torch.nn.BatchNorm1d(out_features)
            self.forward(torch.randn(2, 2) * 2 + 2)  # update the BN stats

        def forward(self, x):
            y = self.linear(x)
            y = self.bn(y)
            y = self.linear(y)
            y = y + y
            return self.bn(y)

    def test_fp32_batch_norm_fusion(self):
        (
            Tester(
                self.ModelLinearBN(2, 2).eval(),
                (torch.randn(2, 2),),
            )
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({self.bn_name: 1})
            .run_method_and_compare_outputs()
        )

    def test_q8_batch_norm_fusion(self):
        (
            Tester(
                self.ModelLinearBN(2, 2).eval(),
                (torch.randn(2, 2),),
            )
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
        not fused with a linear). This is planned, but until implemented, this test ensures
        that we do not partition the standalone batch norm and then fail to lower.
        """

        class BN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)

            def forward(self, x):
                return self.bn(x)

        (
            Tester(BN(), (torch.randn(2, 2),))
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .partition()
            .check_count({self.bn_name: 1})
        )
