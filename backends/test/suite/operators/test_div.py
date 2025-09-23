# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Optional

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x / y


class ModelWithRounding(torch.nn.Module):
    def __init__(self, rounding_mode: Optional[str]):
        super().__init__()
        self.rounding_mode = rounding_mode

    def forward(self, x, y):
        return torch.div(x, y, rounding_mode=self.rounding_mode)


@parameterize_by_dtype
def test_divide_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            (torch.rand(2, 10) * 100).to(dtype),
            (torch.rand(2, 10) * 100 + 0.1).to(
                dtype
            ),  # Adding 0.1 to avoid division by zero
        ),
    )


def test_divide_f32_bcast_first(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(5),
            torch.randn(1, 5, 1, 5).abs()
            + 0.1,  # Using abs and adding 0.1 to avoid division by zero
        ),
    )


def test_divide_f32_bcast_second(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(4, 4, 2, 7),
            torch.randn(2, 7).abs()
            + 0.1,  # Using abs and adding 0.1 to avoid division by zero
        ),
    )


def test_divide_f32_bcast_unary(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (
            torch.randn(5),
            torch.randn(1, 1, 5).abs()
            + 0.1,  # Using abs and adding 0.1 to avoid division by zero
        ),
    )


def test_divide_f32_trunc(test_runner) -> None:
    test_runner.lower_and_run_model(
        ModelWithRounding(rounding_mode="trunc"),
        (
            torch.randn(3, 4) * 10,
            torch.randn(3, 4).abs()
            + 0.1,  # Using abs and adding 0.1 to avoid division by zero
        ),
    )


def test_divide_f32_floor(test_runner) -> None:
    test_runner.lower_and_run_model(
        ModelWithRounding(rounding_mode="floor"),
        (
            torch.randn(3, 4) * 10,
            torch.randn(3, 4).abs()
            + 0.1,  # Using abs and adding 0.1 to avoid division by zero
        ),
    )
