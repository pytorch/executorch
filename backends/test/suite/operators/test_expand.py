# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class ExpandModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.expand(self.shape)


@parameterize_by_dtype
def test_expand_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        ExpandModel(shape=[8, 32]),
        (torch.rand(1, 32).to(dtype),),
    )


def test_expand_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        ExpandModel(shape=[8, 32]),
        (torch.randn(1, 32),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[16, 20]),
        (torch.randn(1, 1),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[4, 1, 32]),
        (torch.randn(1, 32),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[8, 4, 16]),
        (torch.randn(8, 1, 16),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[6, 16, 8]),
        (torch.randn(6, 16, 1),),
    )


def test_expand_keep_original_size(test_runner) -> None:
    test_runner.lower_and_run_model(
        ExpandModel(shape=[8, -1]),
        (torch.randn(1, 32),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[-1, 32]),
        (torch.randn(4, 1),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[-1, 16, -1]),
        (torch.randn(4, 1, 8),),
    )


def test_expand_rank_increase(test_runner) -> None:
    # Test expanding 2D tensor to 3D
    test_runner.lower_and_run_model(
        ExpandModel(shape=[6, 8, 16]),
        (torch.randn(8, 16),),
    )

    # Test expanding 2D tensor to 4D
    test_runner.lower_and_run_model(
        ExpandModel(shape=[3, 4, 8, 16]),
        (torch.randn(8, 16),),
    )


def test_expand_singleton_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        ExpandModel(shape=[512]),
        (torch.randn(1),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[16, 20]),
        (torch.randn(1, 1),),
    )

    test_runner.lower_and_run_model(
        ExpandModel(shape=[8, 32]),
        (torch.randn(32),),
    )
