# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class TruncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.trunc(x)


@parameterize_by_dtype
def test_trunc_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = TruncModel().to(dtype)
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype) * 10 - 5,))


def test_trunc_shapes(test_runner) -> None:
    # Test with different tensor shapes

    # 1D tensor
    test_runner.lower_and_run_model(TruncModel(), (torch.randn(20) * 5,))

    # 2D tensor
    test_runner.lower_and_run_model(TruncModel(), (torch.randn(5, 10) * 5,))

    # 3D tensor
    test_runner.lower_and_run_model(TruncModel(), (torch.randn(3, 4, 5) * 5,))


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_trunc_edge_cases(test_runner) -> None:
    # Test edge cases

    # Integer values (should remain unchanged)
    test_runner.lower_and_run_model(
        TruncModel(),
        (torch.arange(-5, 6).float(),),
        generate_random_test_inputs=False,
    )

    # Values with different fractional parts
    x = torch.tensor(
        [-2.9, -2.5, -2.1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 2.1, 2.5, 2.9]
    )
    test_runner.lower_and_run_model(
        TruncModel(), (x,), generate_random_test_inputs=False
    )

    # Tensor with infinity
    x = torch.tensor([float("inf"), float("-inf"), 1.4, -1.4])
    test_runner.lower_and_run_model(
        TruncModel(), (x,), generate_random_test_inputs=False
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 1.4, -1.4])
    test_runner.lower_and_run_model(
        TruncModel(), (x,), generate_random_test_inputs=False
    )
