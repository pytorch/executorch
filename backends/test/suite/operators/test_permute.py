# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class PermuteModel(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


@parameterize_by_dtype
def test_permute_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[1, 0]),
        (torch.rand(20, 32).to(dtype),),
    )


def test_permute_3d(test_runner) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[2, 0, 1]),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[1, 2, 0]),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[0, 2, 1]),
        (torch.randn(8, 10, 12),),
    )


def test_permute_4d(test_runner) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[3, 2, 1, 0]),
        (torch.randn(4, 6, 8, 10),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[0, 2, 1, 3]),
        (torch.randn(4, 6, 8, 10),),
    )


def test_permute_identity(test_runner) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[0, 1]),
        (torch.randn(20, 32),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[0, 1, 2]),
        (torch.randn(8, 10, 12),),
    )


def test_permute_negative_dims(test_runner) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[-1, -3, -2, -4]),
        (torch.randn(4, 6, 8, 10),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[-4, -2, -3, -1]),
        (torch.randn(4, 6, 8, 10),),
    )


def test_permute_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        PermuteModel(dims=[0]),
        (torch.randn(512),),
    )

    test_runner.lower_and_run_model(
        PermuteModel(dims=[4, 3, 2, 1, 0]),
        (torch.randn(2, 3, 4, 5, 6),),
    )
