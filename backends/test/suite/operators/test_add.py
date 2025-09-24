# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import pytest
import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class ModelAlpha(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        return torch.add(x, y, alpha=self.alpha)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32], ids=lambda s: str(s)[6:]
)
def test_add_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            (torch.rand(2, 10) * 100).to(dtype),
            (torch.rand(2, 10) * 100).to(dtype),
        ),
    )


def test_add_f32_bcast_first(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(5),
            torch.randn(1, 5, 1, 5),
        ),
    )


def test_add_f32_bcast_second(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(4, 4, 2, 7),
            torch.randn(2, 7),
        ),
    )


def test_add_f32_bcast_unary(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(5),
            torch.randn(1, 1, 5),
        ),
    )


def test_add_f32_alpha(test_runner) -> None:
    test_runner.lower_and_run_model(
        ModelAlpha(alpha=2),
        (
            torch.randn(1, 25),
            torch.randn(1, 25),
        ),
    )
