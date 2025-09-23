# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class NegModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.neg(x)


@parameterize_by_dtype
def test_neg_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = NegModel().to(dtype)
    test_runner.lower_and_run_model(
        model,
        (torch.rand(10, 10).to(dtype) * 2 - 1,),
        generate_random_test_inputs=False,
    )


def test_neg_shapes(test_runner) -> None:
    # Test with different tensor shapes

    # 1D tensor
    test_runner.lower_and_run_model(
        NegModel(), (torch.randn(20),), generate_random_test_inputs=False
    )

    # 2D tensor
    test_runner.lower_and_run_model(
        NegModel(), (torch.randn(5, 10),), generate_random_test_inputs=False
    )

    # 3D tensor
    test_runner.lower_and_run_model(
        NegModel(), (torch.randn(3, 4, 5),), generate_random_test_inputs=False
    )


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_neg_edge_cases(test_runner) -> None:
    # Test edge cases

    # Tensor with infinity
    x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
    test_runner.lower_and_run_model(NegModel(), (x,), generate_random_test_inputs=False)

    # Tensor with NaN
    x = torch.tensor([float("nan"), 1.0, -1.0])
    test_runner.lower_and_run_model(NegModel(), (x,), generate_random_test_inputs=False)
