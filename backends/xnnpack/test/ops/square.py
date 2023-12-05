# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSquare(unittest.TestCase):
    class Square(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = torch.square(x)
            return z

    def test_fp32_square(self):
        """
        Note that torch.square maps to aten.pow.Tensor_Scalar. The pow visitor has logic
        to emit the appropriate XNNPACK square op when the exponent is 2.
        """
        inputs = (torch.randn(20),)
        (
            Tester(self.Square(), inputs)
            .export()
            .check_count({"torch.ops.aten.pow.Tensor_Scalar": 1})
            .to_edge()
            .check_count(
                {"executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 1}
            )
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
