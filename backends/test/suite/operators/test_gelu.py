# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.approximate)


@parameterize_by_dtype
def test_gelu_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),))


def test_gelu_f32_single_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(20),))


def test_gelu_f32_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(2, 3, 4, 5),))


def test_gelu_f32_tanh_approximation(test_runner) -> None:
    test_runner.lower_and_run_model(Model(approximate="tanh"), (torch.randn(3, 4, 5),))


def test_gelu_f32_boundary_values(test_runner) -> None:
    # Test with specific values spanning negative and positive ranges
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    test_runner.lower_and_run_model(Model(), (x,))


def test_gelu_f32_tanh_boundary_values(test_runner) -> None:
    # Test tanh approximation with specific values
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    test_runner.lower_and_run_model(Model(approximate="tanh"), (x,))
