# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class StackModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2, x3):
        return torch.stack([x1, x2, x3], dim=self.dim)


@operator_test
class Stack(OperatorTest):
    @dtype_test
    def test_stack_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            StackModel(),
            (
                torch.rand(3, 4).to(dtype),
                torch.rand(3, 4).to(dtype),
                torch.rand(3, 4).to(dtype),
            ),
            flow,
        )

    def test_stack_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            StackModel(dim=0),
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            flow,
        )

        self._test_op(
            StackModel(dim=1),
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            flow,
        )

        self._test_op(
            StackModel(dim=2),
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            flow,
        )

    def test_stack_negative_dim(self, flow: TestFlow) -> None:
        self._test_op(
            StackModel(dim=-1),
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            flow,
        )

        self._test_op(
            StackModel(dim=-2),
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            flow,
        )

    def test_stack_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            StackModel(),
            (
                torch.randn(5),
                torch.randn(5),
                torch.randn(5),
            ),
            flow,
        )

        self._test_op(
            StackModel(),
            (
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
            ),
            flow,
        )
