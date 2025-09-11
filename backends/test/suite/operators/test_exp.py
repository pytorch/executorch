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


class ExpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


@operator_test
class TestExp(OperatorTest):
    @dtype_test
    def test_exp_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = ExpModel().to(dtype)
        # Use smaller range to avoid overflow
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 4 - 2,), flow)

    def test_exp_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(ExpModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(ExpModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(ExpModel(), (torch.randn(3, 4, 5),), flow)

    def test_exp_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(ExpModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(ExpModel(), (x,), flow, generate_random_test_inputs=False)

        # Overflow
        x = torch.tensor([10e10])
        self._test_op(ExpModel(), (x,), flow, generate_random_test_inputs=False)
