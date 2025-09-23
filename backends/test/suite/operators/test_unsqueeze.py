# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class UnsqueezeModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


@parameterize_by_dtype
def test_unsqueeze_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=1),
        (torch.rand(3, 5).to(dtype),),
    )


def test_unsqueeze_basic(test_runner) -> None:
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=1),
        (torch.randn(3, 5),),
    )


def test_unsqueeze_positions(test_runner) -> None:
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=0),
        (torch.randn(3, 5),),
    )

    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=1),
        (torch.randn(3, 5),),
    )

    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=2),
        (torch.randn(3, 5),),
    )


def test_unsqueeze_negative_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=-1),
        (torch.randn(3, 5),),
    )

    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=-2),
        (torch.randn(3, 5),),
    )

    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=-3),
        (torch.randn(3, 5),),
    )


def test_unsqueeze_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=0),
        (torch.randn(5),),
    )
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=1),
        (torch.randn(5),),
    )

    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=0),
        (torch.randn(3, 4, 5),),
    )
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=2),
        (torch.randn(3, 4, 5),),
    )
    test_runner.lower_and_run_model(
        UnsqueezeModel(dim=3),
        (torch.randn(3, 4, 5),),
    )
