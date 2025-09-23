# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class SplitSizeModel(torch.nn.Module):
    def __init__(self, split_size: int, dim: int = 0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.split_size, dim=self.dim)


class SplitSectionsModel(torch.nn.Module):
    def __init__(self, sections: List[int], dim: int = 0):
        super().__init__()
        self.sections = sections
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sections, dim=self.dim)


@parameterize_by_dtype
def test_split_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2),
        (torch.rand(6, 4).to(dtype),),
    )


def test_split_size_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2, dim=0),
        (torch.randn(6, 4),),
    )

    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2, dim=1),
        (torch.randn(4, 6),),
    )

    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2, dim=2),
        (torch.randn(3, 4, 6),),
    )


def test_split_size_uneven(test_runner) -> None:
    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=3),
        (torch.randn(7, 4),),
    )

    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=3, dim=1),
        (torch.randn(4, 7),),
    )


def test_split_sections_dimensions(test_runner) -> None:
    test_runner.lower_and_run_model(
        SplitSectionsModel(sections=[2, 3, 1], dim=0),
        (torch.randn(6, 4),),
    )

    test_runner.lower_and_run_model(
        SplitSectionsModel(sections=[2, 3, 1], dim=1),
        (torch.randn(4, 6),),
    )

    test_runner.lower_and_run_model(
        SplitSectionsModel(sections=[2, 3, 1], dim=2),
        (torch.randn(3, 4, 6),),
    )


def test_split_negative_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2, dim=-1),
        (torch.randn(4, 6),),
    )

    test_runner.lower_and_run_model(
        SplitSizeModel(split_size=2, dim=-2),
        (torch.randn(4, 6),),
    )

    test_runner.lower_and_run_model(
        SplitSectionsModel(sections=[2, 3, 1], dim=-1),
        (torch.randn(4, 6),),
    )
