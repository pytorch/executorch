# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class StackModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2, x3):
        return torch.stack([x1, x2, x3], dim=self.dim)


@parameterize_by_dtype
def test_stack_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        StackModel(),
        (
            torch.rand(3, 4).to(dtype),
            torch.rand(3, 4).to(dtype),
            torch.rand(3, 4).to(dtype),
        ),
    )


def test_stack_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        StackModel(dim=0),
        (
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
    )

    test_runner.lower_and_run_model(
        StackModel(dim=1),
        (
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
    )

    test_runner.lower_and_run_model(
        StackModel(dim=2),
        (
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
    )


def test_stack_negative_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        StackModel(dim=-1),
        (
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
    )

    test_runner.lower_and_run_model(
        StackModel(dim=-2),
        (
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        ),
    )


def test_stack_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        StackModel(),
        (
            torch.randn(5),
            torch.randn(5),
            torch.randn(5),
        ),
    )

    test_runner.lower_and_run_model(
        StackModel(),
        (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        ),
    )
