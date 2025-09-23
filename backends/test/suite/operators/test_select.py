# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class SelectModel(torch.nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return torch.select(x, dim=self.dim, index=self.index)


@parameterize_by_dtype
def test_select_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        SelectModel(dim=0, index=0),
        (torch.rand(3, 4, 5).to(dtype),),
    )


def test_select_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        SelectModel(dim=0, index=1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        SelectModel(dim=1, index=2),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        SelectModel(dim=2, index=3),
        (torch.randn(3, 4, 5),),
    )


def test_select_negative_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        SelectModel(dim=-1, index=2),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        SelectModel(dim=-2, index=1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        SelectModel(dim=-3, index=0),
        (torch.randn(3, 4, 5),),
    )


def test_select_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        SelectModel(dim=0, index=1),
        (torch.randn(3, 4),),
    )

    test_runner.lower_and_run_model(
        SelectModel(dim=1, index=1),
        (torch.randn(2, 3, 4, 5),),
    )
