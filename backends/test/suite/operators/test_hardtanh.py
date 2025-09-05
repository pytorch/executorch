# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import unittest
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class Model(torch.nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.hardtanh(
            x, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace
        )


@operator_test
class TestHardtanh(OperatorTest):
    @dtype_test
    def test_hardtanh_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 4 - 2).to(dtype),), flow)

    def test_hardtanh_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_hardtanh_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    def test_hardtanh_f32_custom_range(self, flow: TestFlow) -> None:
        self._test_op(Model(min_val=-2.0, max_val=2.0), (torch.randn(3, 4, 5),), flow)

    @unittest.skip("In place activations aren't properly defunctionalized yet.")
    def test_hardtanh_f32_inplace(self, flow: TestFlow) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), flow)

    def test_hardtanh_f32_boundary_values(self, flow: TestFlow) -> None:
        # Test with values that span the hardtanh's piecewise regions
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(Model(), (x,), flow)
