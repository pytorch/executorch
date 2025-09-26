# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class SquareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


@parameterize_by_dtype
def test_square_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = SquareModel().to(dtype)
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype) * 2 - 1,))


def test_square_shapes(test_runner) -> None:
    # Test with different tensor shapes

    # 1D tensor
    test_runner.lower_and_run_model(SquareModel(), (torch.randn(20),))

    # 2D tensor
    test_runner.lower_and_run_model(SquareModel(), (torch.randn(5, 10),))

    # 3D tensor
    test_runner.lower_and_run_model(SquareModel(), (torch.randn(3, 4, 5),))


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_square_edge_cases(test_runner) -> None:
    # Test edge cases

    # Tensor with infinity
    x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
    test_runner.lower_and_run_model(
        SquareModel(), (x,), generate_random_test_inputs=False
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 1.0, -1.0])
    test_runner.lower_and_run_model(
        SquareModel(), (x,), generate_random_test_inputs=False
    )

    # Very large values (close to overflow for some dtypes)
    x = torch.tensor([1e10, -1e10])
    test_runner.lower_and_run_model(
        SquareModel(), (x,), generate_random_test_inputs=False
    )

    # Very small values (close to underflow)
    x = torch.tensor([1e-10, -1e-10])
    test_runner.lower_and_run_model(
        SquareModel(), (x,), generate_random_test_inputs=False
    )
