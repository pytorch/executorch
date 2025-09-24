# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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


@parameterize_by_dtype
def test_pow_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = PowModel(2).to(dtype)
    # Use positive values to avoid complex results with fractional powers
    test_runner.lower_and_run_model(
        model,
        (torch.rand(10, 10).to(dtype) + 0.1,),
        generate_random_test_inputs=False,
    )


def test_pow_scalar_exponents(test_runner) -> None:
    # Test with different scalar exponents

    # Power of 0 (should return 1 for all inputs)
    test_runner.lower_and_run_model(
        PowModel(0),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Power of 1 (should return the input unchanged)
    test_runner.lower_and_run_model(
        PowModel(1),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Power of 2 (squaring)
    test_runner.lower_and_run_model(
        PowModel(2),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Power of 3 (cubing)
    test_runner.lower_and_run_model(
        PowModel(3),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Negative power (-1, reciprocal)
    test_runner.lower_and_run_model(
        PowModel(-1),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Fractional power (square root)
    test_runner.lower_and_run_model(
        PowModel(0.5),
        (torch.rand(10, 10) + 0.1,),
        generate_random_test_inputs=False,
    )

    # Large power
    test_runner.lower_and_run_model(
        PowModel(10),
        (torch.rand(10, 10) * 0.5 + 0.5,),
        generate_random_test_inputs=False,
    )


def test_pow_shapes(test_runner) -> None:
    # Test with different tensor shapes
    model = PowModel(2)  # Square the input

    # 1D tensor
    test_runner.lower_and_run_model(
        model, (torch.rand(20) + 0.1,), generate_random_test_inputs=False
    )

    # 2D tensor
    test_runner.lower_and_run_model(
        model, (torch.rand(5, 10) + 0.1,), generate_random_test_inputs=False
    )

    # 3D tensor
    test_runner.lower_and_run_model(
        model, (torch.rand(3, 4, 5) + 0.1,), generate_random_test_inputs=False
    )


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_pow_edge_cases(test_runner) -> None:
    # Test edge cases

    # 0^0 = 1 (by convention)
    x = torch.zeros(1)
    y = torch.zeros(1)
    test_runner.lower_and_run_model(
        PowTensorModel(), (x, y), generate_random_test_inputs=False
    )

    # Tensor with infinity
    x = torch.tensor([float("inf"), 2.0, 3.0])
    y = torch.tensor([2.0, 2.0, 2.0])
    test_runner.lower_and_run_model(
        PowTensorModel(), (x, y), generate_random_test_inputs=False
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 2.0, 3.0])
    y = torch.tensor([2.0, 2.0, 2.0])
    test_runner.lower_and_run_model(
        PowTensorModel(), (x, y), generate_random_test_inputs=False
    )
