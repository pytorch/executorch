# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Log1pModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log1p(x)


@parameterize_by_dtype
def test_log1p_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = Log1pModel().to(dtype)
    # Use values greater than -1 for log1p
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype) * 2 - 0.5,))


def test_log1p_shapes(test_runner) -> None:
    # Test with different tensor shapes

    # 1D tensor
    test_runner.lower_and_run_model(Log1pModel(), (torch.rand(20) * 2 - 0.5,))

    # 2D tensor
    test_runner.lower_and_run_model(Log1pModel(), (torch.rand(5, 10) * 2 - 0.5,))

    # 3D tensor
    test_runner.lower_and_run_model(Log1pModel(), (torch.rand(3, 4, 5) * 2 - 0.5,))


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_log1p_edge_cases(test_runner) -> None:
    # Test edge cases
    # Tensor with infinity
    x = torch.tensor([float("inf"), 0.0, 1.0])
    test_runner.lower_and_run_model(
        Log1pModel(), (x,), generate_random_test_inputs=False
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 0.0, 1.0])
    test_runner.lower_and_run_model(
        Log1pModel(), (x,), generate_random_test_inputs=False
    )
