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


class SqrtModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(x)


@operator_test
class TestSqrt(OperatorTest):
    @dtype_test
    def test_sqrt_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = SqrtModel().to(dtype)
        # Use non-negative values only for sqrt
        self._test_op(model, (torch.rand(10, 10).to(dtype),), flow)

    def test_sqrt_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with non-negative values
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 10,), flow)

    def test_sqrt_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(SqrtModel(), (torch.rand(20),), flow)

        # 2D tensor
        self._test_op(SqrtModel(), (torch.rand(5, 10),), flow)

        # 3D tensor
        self._test_op(SqrtModel(), (torch.rand(3, 4, 5),), flow)

        # 4D tensor
        self._test_op(SqrtModel(), (torch.rand(2, 3, 4, 5),), flow)

        # 5D tensor
        self._test_op(SqrtModel(), (torch.rand(2, 2, 3, 4, 5),), flow)

    def test_sqrt_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small values close to zero
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 0.01,), flow)

        # Values around 1
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 0.2 + 0.9,), flow)
        # Perfect squares
        x = torch.tensor(
            [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]
        )
        self._test_op(SqrtModel(), (x,), flow, generate_random_test_inputs=False)

    def test_sqrt_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Zero tensor
        self._test_op(
            SqrtModel(), (torch.zeros(10, 10),), flow, generate_random_test_inputs=False
        )

        # Tensor with specific values
        x = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.25, 0.5, 0.01])
        self._test_op(SqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), 1.0, 4.0])
        self._test_op(SqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, 4.0])
        self._test_op(SqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Values very close to zero
        x = torch.tensor([1e-10, 1e-20, 1e-30])
        self._test_op(SqrtModel(), (x,), flow, generate_random_test_inputs=False)

    def test_sqrt_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            SqrtModel(), (torch.tensor([0.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            SqrtModel(), (torch.tensor([1.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            SqrtModel(), (torch.tensor([4.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            SqrtModel(),
            (torch.tensor([0.25]),),
            flow,
            generate_random_test_inputs=False,
        )
