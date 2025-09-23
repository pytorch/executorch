# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Optional

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class ArgmaxModel(torch.nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)


@parameterize_by_dtype
def test_argmax_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        ArgmaxModel().to(dtype),
        (torch.rand(10, 10).to(dtype),),
    )


def test_argmax_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=0),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=0),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=2),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=-1),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=-2),
        (torch.randn(3, 4, 5),),
    )


def test_argmax_keepdim(test_runner) -> None:
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=0, keepdim=True),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1, keepdim=True),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1, keepdim=True),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(dim=2, keepdim=True),
        (torch.randn(2, 3, 4, 5),),
    )


def test_argmax_shapes(test_runner) -> None:
    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.randn(20),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.randn(5, 10),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.randn(3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.randn(2, 3, 4, 5),),
    )

    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.randn(2, 2, 3, 4, 5),),
    )


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_argmax_edge_cases(test_runner) -> None:
    x = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("inf")]])
    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=0),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1),
        (x,),
        generate_random_test_inputs=False,
    )

    x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=0),
        (x,),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ArgmaxModel(dim=1),
        (x,),
        generate_random_test_inputs=False,
    )

    x = torch.tensor([5.0])
    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (x,),
    )


def test_argmax_scalar(test_runner) -> None:
    test_runner.lower_and_run_model(
        ArgmaxModel(),
        (torch.tensor([5.0]),),
    )
