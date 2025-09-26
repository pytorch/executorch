# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class CatModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=self.dim)


@parameterize_by_dtype
def test_cat_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        CatModel(),
        (
            torch.rand(8, 32).to(dtype),
            torch.rand(12, 32).to(dtype),
            torch.rand(16, 32).to(dtype),
        ),
    )


def test_cat_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        CatModel(dim=0),
        (
            torch.randn(8, 32),
            torch.randn(12, 32),
            torch.randn(16, 32),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=1),
        (
            torch.randn(16, 8),
            torch.randn(16, 12),
            torch.randn(16, 16),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=2),
        (
            torch.randn(4, 8, 4),
            torch.randn(4, 8, 8),
            torch.randn(4, 8, 12),
        ),
    )


def test_cat_negative_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        CatModel(dim=-1),
        (
            torch.randn(16, 8),
            torch.randn(16, 12),
            torch.randn(16, 16),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=-2),
        (
            torch.randn(8, 32),
            torch.randn(12, 32),
            torch.randn(16, 32),
        ),
    )


def test_cat_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        CatModel(),
        (
            torch.randn(128),
            torch.randn(256),
            torch.randn(384),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=0),
        (
            torch.randn(4, 8, 16),
            torch.randn(8, 8, 16),
            torch.randn(12, 8, 16),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=1),
        (
            torch.randn(8, 4, 16),
            torch.randn(8, 8, 16),
            torch.randn(8, 12, 16),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=2),
        (
            torch.randn(8, 12, 4),
            torch.randn(8, 12, 8),
            torch.randn(8, 12, 12),
        ),
    )


def test_cat_broadcast(test_runner) -> None:
    test_runner.lower_and_run_model(
        CatModel(dim=0),
        (
            torch.randn(2, 16, 32),
            torch.randn(4, 16, 32),
            torch.randn(6, 16, 32),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=1),
        (
            torch.randn(8, 8, 16),
            torch.randn(8, 16, 16),
            torch.randn(8, 24, 16),
        ),
    )

    test_runner.lower_and_run_model(
        CatModel(dim=2),
        (
            torch.randn(4, 16, 8),
            torch.randn(4, 16, 16),
            torch.randn(4, 16, 24),
        ),
    )


def test_cat_same_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        CatModel(),
        (
            torch.randn(8, 32),
            torch.randn(8, 32),
            torch.randn(8, 32),
        ),
    )
