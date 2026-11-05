# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestAddmm(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class AddmmWithScales(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))
            self.bias = torch.nn.Parameter(torch.randn(8))

        def forward(self, x):
            return (
                # partitioned
                torch.addmm(self.bias, x, self.weight),
                # not partitioned, since beta or alpha is not 1
                torch.addmm(self.bias, x, self.weight, beta=2.0),
                torch.addmm(self.bias, x, self.weight, alpha=0.5),
            )

    def test_fp32_addmm_with_scales(self):
        (
            Tester(self.AddmmWithScales(), (torch.randn(6, 16),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_addmm_default": 2})
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
