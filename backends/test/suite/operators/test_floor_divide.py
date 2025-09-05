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


class FloorDivideModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


@operator_test
class TestFloorDivide(OperatorTest):
    @dtype_test
    def test_floor_divide_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = FloorDivideModel().to(dtype)
        # Use values that won't cause division by zero
        x = torch.randint(-100, 100, (10, 10)).to(dtype)
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

    def test_floor_divide_scalar_divisors(self, flow: TestFlow) -> None:
        # Test with different scalar divisors as tensors

        # Positive divisor
        x = torch.randint(-100, 100, (10, 10))
        y = torch.full_like(x, 3)  # Divisor of 3
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Negative divisor
        x = torch.randint(-100, 100, (10, 10))
        y = torch.full_like(x, -2)  # Divisor of -2
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Fractional divisor
        x = torch.randint(-100, 100, (10, 10)).float()
        y = torch.full_like(x, 2.5)  # Divisor of 2.5
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Large divisor
        x = torch.randint(-1000, 1000, (10, 10))
        y = torch.full_like(x, 100)  # Divisor of 100
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Small divisor
        x = torch.randint(-100, 100, (10, 10)).float()
        y = torch.full_like(x, 0.5)  # Divisor of 0.5
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

    def test_floor_divide_tensor_divisors(self, flow: TestFlow) -> None:
        # Test with tensor divisors

        # Constant divisor tensor
        x = torch.randint(-100, 100, (10, 10))
        y = torch.full_like(x, 2)  # All elements are 2
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Random divisor tensor (non-zero)
        x = torch.randint(-100, 100, (10, 10))
        y = torch.randint(1, 10, (10, 10))  # Positive divisors
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Mixed positive and negative divisors
        x = torch.randint(-100, 100, (10, 10))
        y = torch.randint(-10, 10, (10, 10))
        # Replace zeros to avoid division by zero
        y[y == 0] = 1
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Broadcasting: scalar dividend, tensor divisor
        x = torch.tensor([10])
        y = torch.arange(1, 5)  # [1, 2, 3, 4]
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

        # Broadcasting: tensor dividend, scalar divisor
        x = torch.arange(-10, 10)
        y = torch.tensor([2])
        self._test_op(
            FloorDivideModel(), (x, y), flow, generate_random_test_inputs=False
        )

    def test_floor_divide_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes
        model = FloorDivideModel()

        # 1D tensor
        x = torch.randint(-100, 100, (20,))
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # 2D tensor
        x = torch.randint(-100, 100, (5, 10))
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # 3D tensor
        x = torch.randint(-100, 100, (3, 4, 5))
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # 4D tensor
        x = torch.randint(-100, 100, (2, 3, 4, 5))
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # 5D tensor
        x = torch.randint(-100, 100, (2, 2, 3, 4, 5))
        y = torch.full_like(x, 2)  # Divisor of 2
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

    def test_floor_divide_values(self, flow: TestFlow) -> None:
        # Test with different value ranges
        model = FloorDivideModel()

        # Test with specific dividend values
        x = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

        # Divide by 2
        y = torch.tensor([2]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Divide by -2
        y = torch.tensor([-2]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Divide by 3
        y = torch.tensor([3]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Divide by -3
        y = torch.tensor([-3]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Test with floating point values
        x = torch.tensor(
            [-3.8, -3.5, -3.2, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 3.2, 3.5, 3.8]
        )

        # Divide by 2.0
        y = torch.tensor([2.0]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Divide by -2.0
        y = torch.tensor([-2.0]).expand_as(x).clone()
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_floor_divide_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases
        model = FloorDivideModel()

        # Zero dividend
        x = torch.zeros(10)
        y = torch.full_like(x, 2)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Division with remainder
        x = torch.tensor([1, 3, 5, 7, 9])
        y = torch.full_like(x, 2)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 10.0, -10.0])
        y = torch.full_like(x, 2)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 10.0, -10.0])
        y = torch.full_like(x, 2)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Very large values
        x = torch.tensor([1e10, -1e10])
        y = torch.full_like(x, 3)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)

        # Very small values
        x = torch.tensor([1e-10, -1e-10])
        y = torch.full_like(x, 2)
        self._test_op(model, (x, y), flow, generate_random_test_inputs=False)
