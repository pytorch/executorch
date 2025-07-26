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


class CatModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=self.dim)


@operator_test
class Cat(OperatorTest):
    @dtype_test
    def test_cat_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            CatModel(),
            (
                torch.rand(2, 3).to(dtype),
                torch.rand(3, 3).to(dtype),
                torch.rand(4, 3).to(dtype),
            ),
            flow,
        )

    def test_cat_basic(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(),
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ),
            flow,
        )

    def test_cat_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(dim=0),
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=1),
            (
                torch.randn(3, 2),
                torch.randn(3, 3),
                torch.randn(3, 4),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=2),
            (
                torch.randn(2, 3, 1),
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 3),
            ),
            flow,
        )

    def test_cat_negative_dim(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(dim=-1),
            (
                torch.randn(3, 2),
                torch.randn(3, 3),
                torch.randn(3, 4),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=-2),
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ),
            flow,
        )

    def test_cat_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(),
            (
                torch.randn(2),
                torch.randn(3),
                torch.randn(4),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=0),
            (
                torch.randn(1, 3, 4),
                torch.randn(2, 3, 4),
                torch.randn(3, 3, 4),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=1),
            (
                torch.randn(2, 1, 4),
                torch.randn(2, 2, 4),
                torch.randn(2, 3, 4),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=2),
            (
                torch.randn(2, 3, 1),
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 3),
            ),
            flow,
        )

    def test_cat_same_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(),
            (
                torch.randn(2, 3),
                torch.randn(2, 3),
                torch.randn(2, 3),
            ),
            flow,
        )
