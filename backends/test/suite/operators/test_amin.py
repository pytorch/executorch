# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import List, Optional, Tuple, Union

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class AminModel(torch.nn.Module):
    def __init__(
        self,
        dim: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdim: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return torch.amin(x, keepdim=self.keepdim)
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


@parameterize_by_dtype
def test_amin_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        AminModel().to(dtype),
        (torch.rand(10, 10).to(dtype),),
    )


def test_amin_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=1),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=2),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=1),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=-1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=-2),
        (torch.randn(3, 4, 5),),
    )


def test_amin_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        AminModel(dim=(0, 1)),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(0, 2)),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(1, 2)),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(1, 3)),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(0, 2)),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(-1, -3)),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(0, 1, 2, 3)),
        (torch.randn(2, 3, 4, 5),),
    )


def test_amin_keepdim(test_runner) -> None:
    test_runner.lower_and_run_model(
        AminModel(dim=0, keepdim=True),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=1, keepdim=True),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=1, keepdim=True),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=2, keepdim=True),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(dim=(1, 2), keepdim=True),
        (torch.randn(3, 4, 5),),
    )


def test_amin_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        AminModel(),
        (torch.randn(20),),
    )
    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (torch.randn(20),),
    )

    test_runner.lower_and_run_model(
        AminModel(),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        AminModel(),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        AminModel(),
        (torch.randn(2, 2, 3, 4, 5),),
    )


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_amin_edge_cases(test_runner) -> None:
    x = torch.tensor([[1.0, float("-inf"), 3.0], [4.0, 5.0, float("-inf")]])
    test_runner.lower_and_run_model(
        AminModel(),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        AminModel(dim=1),
        (x,),
        generate_random_test_inputs=False,
    )

    x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    test_runner.lower_and_run_model(
        AminModel(),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        AminModel(dim=1),
        (x,),
        generate_random_test_inputs=False,
    )


def test_amin_scalar(test_runner) -> None:
    test_runner.lower_and_run_model(
        AminModel(),
        (torch.tensor([5.0]),),
    )
    test_runner.lower_and_run_model(
        AminModel(dim=0),
        (torch.tensor([5.0]),),
    )
