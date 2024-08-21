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

    def _test_prelu(self, module, inputs):
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.prelu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten__prelu_kernel_default"]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("XNNPACK Expects FP16 inputs but FP32 weights")
    def _test_fp16_prelu(self):
        module = self.PReLU().to(torch.float16)
        inputs = (torch.randn(1, 5, 3, 2).to(torch.float16),)
        self._test_prelu(module, inputs)

    def test_fp32_prelu(self):
        module = self.PReLU()
        inputs = (torch.randn(1, 5, 3, 2),)
        self._test_prelu(module, inputs)
