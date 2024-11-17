# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestPow(unittest.TestCase):
    class Pow(torch.nn.Module):
        def __init__(self, exp):
            super().__init__()
            self.exp = exp

        def forward(self, x):
            z = torch.pow(x, self.exp)
            return z

    def _test_pow2(self, inputs):
        for legacy in (True, False):
            tester = Tester(self.Pow(2), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.pow.Tensor_Scalar": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_fp16_pow2(self):
        inputs = (torch.randn(20).to(torch.float16),)
        self._test_pow2(inputs)

    def test_fp32_pow2(self):
        inputs = (torch.randn(20),)
        self._test_pow2(inputs)

    def test_fp32_pow_unsupported(self):
        """

        The XNNPACK backend does not support arbitrary powers. Power of two (square)
        is handled by the square op. This test verifies that the partitioner does not
        attempt to delegate other powers.
        """

        inputs = (torch.randn(5),)
        for legacy in (True, False):
            tester = Tester(self.Pow(3), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.pow.Tensor_Scalar": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_not(["torch.ops.higher_order.executorch_call_delegate"])
