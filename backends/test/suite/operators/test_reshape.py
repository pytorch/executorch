# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class ReshapeModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


@parameterize_by_dtype
def test_reshape_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        ReshapeModel(shape=[3, 5]),
        (torch.rand(15).to(dtype),),
    )


def test_reshape_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        ReshapeModel(shape=[3, 5]),
        (torch.randn(15),),
    )

    test_runner.lower_and_run_model(
        ReshapeModel(shape=[20]),
        (torch.randn(4, 5),),
    )

    test_runner.lower_and_run_model(
        ReshapeModel(shape=[2, 2, 5]),
        (torch.randn(4, 5),),
    )

    test_runner.lower_and_run_model(
        ReshapeModel(shape=[6, 4]),
        (torch.randn(3, 2, 4),),
    )


def test_reshape_inferred_dimension(test_runner) -> None:
    test_runner.lower_and_run_model(
        ReshapeModel(shape=[3, -1]),
        (torch.randn(15),),
    )

    test_runner.lower_and_run_model(
        ReshapeModel(shape=[-1, 5]),
        (torch.randn(15),),
    )

    test_runner.lower_and_run_model(
        ReshapeModel(shape=[2, -1, 3]),
        (torch.randn(24),),
    )
