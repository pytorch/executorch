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
                torch.rand(8, 32).to(dtype),
                torch.rand(12, 32).to(dtype),
                torch.rand(16, 32).to(dtype),
            ),
            flow,
        )

    def test_cat_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(dim=0),
            (
                torch.randn(8, 32),
                torch.randn(12, 32),
                torch.randn(16, 32),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=1),
            (
                torch.randn(16, 8),
                torch.randn(16, 12),
                torch.randn(16, 16),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=2),
            (
                torch.randn(4, 8, 4),
                torch.randn(4, 8, 8),
                torch.randn(4, 8, 12),
            ),
            flow,
        )

    def test_cat_negative_dim(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(dim=-1),
            (
                torch.randn(16, 8),
                torch.randn(16, 12),
                torch.randn(16, 16),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=-2),
            (
                torch.randn(8, 32),
                torch.randn(12, 32),
                torch.randn(16, 32),
            ),
            flow,
        )

    def test_cat_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(),
            (
                torch.randn(128),
                torch.randn(256),
                torch.randn(384),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=0),
            (
                torch.randn(4, 8, 16),
                torch.randn(8, 8, 16),
                torch.randn(12, 8, 16),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=1),
            (
                torch.randn(8, 4, 16),
                torch.randn(8, 8, 16),
                torch.randn(8, 12, 16),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=2),
            (
                torch.randn(8, 12, 4),
                torch.randn(8, 12, 8),
                torch.randn(8, 12, 12),
            ),
            flow,
        )

    def test_cat_broadcast(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(dim=0),
            (
                torch.randn(2, 16, 32),
                torch.randn(4, 16, 32),
                torch.randn(6, 16, 32),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=1),
            (
                torch.randn(8, 8, 16),
                torch.randn(8, 16, 16),
                torch.randn(8, 24, 16),
            ),
            flow,
        )

        self._test_op(
            CatModel(dim=2),
            (
                torch.randn(4, 16, 8),
                torch.randn(4, 16, 16),
                torch.randn(4, 16, 24),
            ),
            flow,
        )

    def test_cat_same_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            CatModel(),
            (
                torch.randn(8, 32),
                torch.randn(8, 32),
                torch.randn(8, 32),
            ),
            flow,
        )
