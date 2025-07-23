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


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x - y


class ModelAlpha(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        return torch.sub(x, y, alpha=self.alpha)


@operator_test
class Subtract(OperatorTest):
    @dtype_test
    def test_subtract_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model(),
            (
                (torch.rand(2, 10) * 100).to(dtype),
                (torch.rand(2, 10) * 100).to(dtype),
            ),
            flow,
        )

    def test_subtract_f32_bcast_first(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 5, 1, 5),
            ),
            flow,
        )

    def test_subtract_f32_bcast_second(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(4, 4, 2, 7),
                torch.randn(2, 7),
            ),
            flow,
        )

    def test_subtract_f32_bcast_unary(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 1, 5),
            ),
            flow,
        )

    def test_subtract_f32_alpha(self, flow: TestFlow) -> None:
        self._test_op(
            ModelAlpha(alpha=2),
            (
                torch.randn(1, 25),
                torch.randn(1, 25),
            ),
            flow,
        )
