# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class TransposeModel(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


@parameterize_by_dtype
def test_transpose_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=1),
        (torch.rand(20, 32).to(dtype),),
    )


def test_transpose_basic(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=1),
        (torch.randn(20, 32),),
    )


def test_transpose_3d(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=1),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=2),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=1, dim1=2),
        (torch.randn(8, 10, 12),),
    )


def test_transpose_4d(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=3),
        (torch.randn(4, 6, 8, 10),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=1, dim1=2),
        (torch.randn(4, 6, 8, 10),),
    )


def test_transpose_identity(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=0),
        (torch.randn(20, 32),),
    )
    test_runner.lower_and_run_model(
        TransposeModel(dim0=1, dim1=1),
        (torch.randn(20, 32),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=0),
        (torch.randn(8, 10, 12),),
    )
    test_runner.lower_and_run_model(
        TransposeModel(dim0=1, dim1=1),
        (torch.randn(8, 10, 12),),
    )
    test_runner.lower_and_run_model(
        TransposeModel(dim0=2, dim1=2),
        (torch.randn(8, 10, 12),),
    )


def test_transpose_negative_dims(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=-3, dim1=-1),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=-2, dim1=-1),
        (torch.randn(8, 10, 12),),
    )


def test_transpose_different_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=1),
        (torch.randn(20, 32),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=2),
        (torch.randn(8, 10, 12),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=1, dim1=3),
        (torch.randn(4, 6, 8, 10),),
    )

    test_runner.lower_and_run_model(
        TransposeModel(dim0=0, dim1=4),
        (torch.randn(2, 3, 4, 5, 6),),
    )
