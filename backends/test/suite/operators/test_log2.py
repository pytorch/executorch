# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Log2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log2(x)


@parameterize_by_dtype
def test_log2_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = Log2Model().to(dtype)
    # Use positive values only for log2
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype) + 0.01,))


def test_log2_shapes(test_runner) -> None:
    # Test with different tensor shapes

    # 1D tensor
    test_runner.lower_and_run_model(Log2Model(), (torch.rand(20) + 0.01,))

    # 2D tensor
    test_runner.lower_and_run_model(Log2Model(), (torch.rand(5, 10) + 0.01,))

    # 3D tensor
    test_runner.lower_and_run_model(Log2Model(), (torch.rand(3, 4, 5) + 0.01,))


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_log2_edge_cases(test_runner) -> None:
    # Test edge cases
    # Tensor with infinity
    x = torch.tensor([float("inf"), 1.0, 2.0])
    test_runner.lower_and_run_model(
        Log2Model(), (x,), generate_random_test_inputs=False
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 1.0, 2.0])
    test_runner.lower_and_run_model(
        Log2Model(), (x,), generate_random_test_inputs=False
    )
