# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.glu(x, dim=self.dim)


@parameterize_by_dtype
def test_glu_dtype(test_runner, dtype) -> None:
    # Input must have even number of elements in the specified dimension
    test_runner.lower_and_run_model(Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),))


def test_glu_f32_dim_last(test_runner) -> None:
    # Default dim is -1 (last dimension)
    test_runner.lower_and_run_model(Model(), (torch.randn(3, 4, 6),))


def test_glu_f32_dim_first(test_runner) -> None:
    # Test with dim=0 (first dimension)
    test_runner.lower_and_run_model(Model(dim=0), (torch.randn(4, 3, 5),))


def test_glu_f32_dim_middle(test_runner) -> None:
    # Test with dim=1 (middle dimension)
    test_runner.lower_and_run_model(Model(dim=1), (torch.randn(3, 8, 5),))


def test_glu_f32_boundary_values(test_runner) -> None:
    # Test with specific values spanning negative and positive ranges
    # Input must have even number of elements in the specified dimension
    x = torch.tensor([[-10.0, -5.0, -1.0, 0.0], [1.0, 5.0, 10.0, -2.0]])
    test_runner.lower_and_run_model(Model(dim=1), (x,))
