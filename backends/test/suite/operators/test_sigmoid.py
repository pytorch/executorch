# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.sigmoid(x)


@operator_test
class TestSigmoid(OperatorTest):
    @dtype_test
    def test_sigmoid_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),), flow)

    def test_sigmoid_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_sigmoid_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    def test_sigmoid_f32_boundary_values(self, flow: TestFlow) -> None:
        # Test with specific values spanning negative and positive ranges
        x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        self._test_op(Model(), (x,), flow)
