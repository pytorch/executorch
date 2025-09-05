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
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.relu(x, self.inplace)


@operator_test
class TestReLU(OperatorTest):
    @dtype_test
    def test_relu_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 100).to(dtype),), flow)

    def test_relu_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_relu_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    @unittest.skip("In place activations aren't properly defunctionalized yet.")
    def test_relu_f32_inplace(self, flow: TestFlow) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), flow)
