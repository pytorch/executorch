# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class SquareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


@operator_test
class TestSquare(OperatorTest):
    @dtype_test
    def test_square_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = SquareModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), flow)

    def test_square_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(SquareModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(SquareModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(SquareModel(), (torch.randn(3, 4, 5),), flow)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_square_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(SquareModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(SquareModel(), (x,), flow, generate_random_test_inputs=False)

        # Very large values (close to overflow for some dtypes)
        x = torch.tensor([1e10, -1e10])
        self._test_op(SquareModel(), (x,), flow, generate_random_test_inputs=False)

        # Very small values (close to underflow)
        x = torch.tensor([1e-10, -1e-10])
        self._test_op(SquareModel(), (x,), flow, generate_random_test_inputs=False)
