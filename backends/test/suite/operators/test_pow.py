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


class PowModel(torch.nn.Module):
    def __init__(self, exponent=None):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        if self.exponent is not None:
            return torch.pow(x, self.exponent)
        return torch.pow(x, 2)  # Default to squaring if no exponent provided


class PowTensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


@operator_test
class TestPow(OperatorTest):
    @dtype_test
    def test_pow_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = PowModel(2).to(dtype)
        # Use positive values to avoid complex results with fractional powers
        self._test_op(
            model,
            (torch.rand(10, 10).to(dtype) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_pow_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters (squaring)
        self._test_op(
            PowModel(),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_pow_scalar_exponents(self, flow: TestFlow) -> None:
        # Test with different scalar exponents

        # Power of 0 (should return 1 for all inputs)
        self._test_op(
            PowModel(0),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Power of 1 (should return the input unchanged)
        self._test_op(
            PowModel(1),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Power of 2 (squaring)
        self._test_op(
            PowModel(2),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Power of 3 (cubing)
        self._test_op(
            PowModel(3),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Negative power (-1, reciprocal)
        self._test_op(
            PowModel(-1),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Fractional power (square root)
        self._test_op(
            PowModel(0.5),
            (torch.rand(10, 10) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Large power
        self._test_op(
            PowModel(10),
            (torch.rand(10, 10) * 0.5 + 0.5,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_pow_tensor_exponents(self, flow: TestFlow) -> None:
        # Test with tensor exponents

        # Constant exponent tensor
        x = torch.rand(10, 10) + 0.1
        y = torch.full_like(x, 2.0)  # All elements are 2.0
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Random exponent tensor (positive values)
        x = torch.rand(10, 10) + 0.1
        y = torch.rand(10, 10) * 3  # Random values between 0 and 3
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Mixed positive and negative exponents
        x = torch.rand(10, 10) + 0.1
        y = torch.randn(10, 10)  # Random values with both positive and negative
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Broadcasting: scalar base, tensor exponent
        x = torch.tensor([2.0])
        y = torch.arange(1, 5).float()  # [1, 2, 3, 4]
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Broadcasting: tensor base, scalar exponent
        x = torch.arange(1, 5).float()  # [1, 2, 3, 4]
        y = torch.tensor([2.0])
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

    def test_pow_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes
        model = PowModel(2)  # Square the input

        # 1D tensor
        self._test_op(
            model, (torch.rand(20) + 0.1,), flow, generate_random_test_inputs=False
        )

        # 2D tensor
        self._test_op(
            model, (torch.rand(5, 10) + 0.1,), flow, generate_random_test_inputs=False
        )

        # 3D tensor
        self._test_op(
            model, (torch.rand(3, 4, 5) + 0.1,), flow, generate_random_test_inputs=False
        )

        # 4D tensor
        self._test_op(
            model,
            (torch.rand(2, 3, 4, 5) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # 5D tensor
        self._test_op(
            model,
            (torch.rand(2, 2, 3, 4, 5) + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_pow_values(self, flow: TestFlow) -> None:
        # Test with different value ranges
        model = PowModel(2)  # Square the input

        # Small values
        self._test_op(
            model,
            (torch.rand(10, 10) * 0.01 + 0.01,),
            flow,
            generate_random_test_inputs=False,
        )

        # Values around 1
        self._test_op(
            model,
            (torch.rand(10, 10) * 0.2 + 0.9,),
            flow,
            generate_random_test_inputs=False,
        )

        # Medium values
        self._test_op(
            model,
            (torch.rand(10, 10) * 10 + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

        # Large values (use smaller exponent to avoid overflow)
        model = PowModel(0.5)  # Square root to avoid overflow
        self._test_op(
            model,
            (torch.rand(10, 10) * 1000 + 0.1,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_pow_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # 0^0 = 1 (by convention)
        x = torch.zeros(1)
        y = torch.zeros(1)
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # x^0 = 1 for any x
        x = torch.randn(10)
        y = torch.zeros(10)
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # 0^y = 0 for y > 0
        x = torch.zeros(5)
        y = torch.arange(1, 6).float()
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # 1^y = 1 for any y
        x = torch.ones(10)
        y = torch.randn(10)
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), 2.0, 3.0])
        y = torch.tensor([2.0, 2.0, 2.0])
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 2.0, 3.0])
        y = torch.tensor([2.0, 2.0, 2.0])
        self._test_op(PowTensorModel(), (x, y), flow, generate_random_test_inputs=False)

    def test_pow_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        model = PowModel(2)  # Square the input
        self._test_op(
            model, (torch.tensor([2.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            model, (torch.tensor([0.5]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            model, (torch.tensor([0.0]),), flow, generate_random_test_inputs=False
        )
