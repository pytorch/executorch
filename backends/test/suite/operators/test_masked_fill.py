# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Union

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class MaskedFillModel(torch.nn.Module):
    def __init__(self, value: Union[float, int]):
        super().__init__()
        self.value = value

    def forward(self, x, mask):
        return x.masked_fill(mask, self.value)


@parameterize_by_dtype
def test_masked_fill_dtype(test_runner, dtype) -> None:
    mask = torch.randint(0, 2, (16, 32), dtype=torch.bool)
    test_runner.lower_and_run_model(
        MaskedFillModel(value=0.0),
        (
            torch.rand(16, 32).to(dtype),
            mask,
        ),
    )


def test_masked_fill_different_values(test_runner) -> None:
    mask = torch.randint(0, 2, (16, 32), dtype=torch.bool)

    test_runner.lower_and_run_model(
        MaskedFillModel(value=5.0),
        (
            torch.randn(16, 32),
            mask,
        ),
    )

    test_runner.lower_and_run_model(
        MaskedFillModel(value=-5.0),
        (
            torch.randn(16, 32),
            mask,
        ),
    )

    test_runner.lower_and_run_model(
        MaskedFillModel(value=1),
        (
            torch.randn(16, 32),
            mask,
        ),
    )


def test_masked_fill_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        MaskedFillModel(value=0.0),
        (
            torch.randn(512),
            torch.randint(0, 2, (512,), dtype=torch.bool),
        ),
    )

    test_runner.lower_and_run_model(
        MaskedFillModel(value=0.0),
        (
            torch.randn(4, 8, 16),
            torch.randint(0, 2, (4, 8, 16), dtype=torch.bool),
        ),
    )


def test_masked_fill_broadcast(test_runner) -> None:
    test_runner.lower_and_run_model(
        MaskedFillModel(value=0.0),
        (
            torch.randn(16, 32),
            torch.randint(0, 2, (32,), dtype=torch.bool),
        ),
    )
