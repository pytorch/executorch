# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class ClampModel(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


@parameterize_by_dtype
def test_clamp_dtype(test_runner, dtype) -> None:
    # Test with different dtypes
    model = ClampModel(min_val=-0.5, max_val=0.5).to(dtype)
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype) * 2 - 1,))


def test_clamp_min_only(test_runner) -> None:
    # Test with only min value specified
    test_runner.lower_and_run_model(ClampModel(min_val=0.0), (torch.randn(10, 10),))


def test_clamp_max_only(test_runner) -> None:
    # Test with only max value specified
    test_runner.lower_and_run_model(ClampModel(max_val=0.0), (torch.randn(10, 10),))


def test_clamp_shapes(test_runner) -> None:
    # Test with different tensor shapes
    model = ClampModel(min_val=-1.0, max_val=1.0)

    # 1D tensor
    test_runner.lower_and_run_model(model, (torch.randn(20),))

    # 2D tensor
    test_runner.lower_and_run_model(model, (torch.randn(5, 10),))

    # 3D tensor
    test_runner.lower_and_run_model(model, (torch.randn(3, 4, 5),))


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_clamp_edge_cases(test_runner) -> None:
    # Test edge cases

    # Min equals max
    test_runner.lower_and_run_model(
        ClampModel(min_val=0.0, max_val=0.0), (torch.randn(10, 10),)
    )

    # Tensor with infinity
    x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
    test_runner.lower_and_run_model(
        ClampModel(min_val=-2.0, max_val=2.0),
        (x,),
        generate_random_test_inputs=False,
    )

    # Tensor with NaN
    x = torch.tensor([float("nan"), 1.0, -1.0])
    test_runner.lower_and_run_model(
        ClampModel(min_val=-2.0, max_val=2.0),
        (x,),
        generate_random_test_inputs=False,
    )
