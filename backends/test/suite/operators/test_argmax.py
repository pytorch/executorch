# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Optional

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class ArgmaxModel(torch.nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)


@operator_test
class Argmax(OperatorTest):
    @dtype_test
    def test_argmax_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            ArgmaxModel().to(dtype),
            (torch.rand(10, 10).to(dtype),),
            flow,
        )

    def test_argmax_dim(self, flow: TestFlow) -> None:
        self._test_op(
            ArgmaxModel(dim=0),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=1),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=0),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=2),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=1),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=-1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=-2),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_argmax_keepdim(self, flow: TestFlow) -> None:
        self._test_op(
            ArgmaxModel(dim=0, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=1, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=1, keepdim=True),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(dim=2, keepdim=True),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

    def test_argmax_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            ArgmaxModel(),
            (torch.randn(20),),
            flow,
        )

        self._test_op(
            ArgmaxModel(),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            ArgmaxModel(),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            ArgmaxModel(),
            (torch.randn(2, 2, 3, 4, 5),),
            flow,
        )

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_argmax_edge_cases(self, flow: TestFlow) -> None:
        x = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("inf")]])
        self._test_op(
            ArgmaxModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            ArgmaxModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            ArgmaxModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        self._test_op(
            ArgmaxModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            ArgmaxModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            ArgmaxModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        x = torch.tensor([5.0])
        self._test_op(
            ArgmaxModel(),
            (x,),
            flow,
        )

    def test_argmax_scalar(self, flow: TestFlow) -> None:
        self._test_op(
            ArgmaxModel(),
            (torch.tensor([5.0]),),
            flow,
        )
