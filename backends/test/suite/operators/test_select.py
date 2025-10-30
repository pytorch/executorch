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


class SelectModel(torch.nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return torch.select(x, dim=self.dim, index=self.index)


@operator_test
class Select(OperatorTest):
    @dtype_test
    def test_select_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            SelectModel(dim=0, index=0),
            (torch.rand(3, 4, 5).to(dtype),),
            flow,
        )

    def test_select_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            SelectModel(dim=0, index=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            SelectModel(dim=1, index=2),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            SelectModel(dim=2, index=3),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_select_negative_dim(self, flow: TestFlow) -> None:
        self._test_op(
            SelectModel(dim=-1, index=2),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            SelectModel(dim=-2, index=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            SelectModel(dim=-3, index=0),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_select_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            SelectModel(dim=0, index=1),
            (torch.randn(3, 4),),
            flow,
        )

        self._test_op(
            SelectModel(dim=1, index=1),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )
