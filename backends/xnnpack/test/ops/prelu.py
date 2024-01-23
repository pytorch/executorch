# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestPrelu(unittest.TestCase):
    class PReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = torch.nn.PReLU(num_parameters=5, init=0.2)

        def forward(self, x):
            a = self.prelu(x)
            return a

    @unittest.skip("T158653285 - Missing recomposition for PReLU")
    def test_fp32_prelu(self):
        inputs = (torch.randn(1, 5, 3, 2),)
        (
            Tester(self.PReLU(), inputs)
            .export()
            .check_count({"torch.ops.aten._prelu_kernel.default": 1})
            .to_edge()
            .check_count(
                {"executorch_exir_dialects_edge__ops_aten__prelu_kernel_default": 1}
            )
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten__prelu_kernel_default"]
            )
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
