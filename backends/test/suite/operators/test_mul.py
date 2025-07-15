# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Callable

import torch

from executorch.backends.test.suite import dtype_test, operator_test, OperatorTest


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@operator_test
class Multiply(OperatorTest):
    @dtype_test
    def test_multiply_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                (torch.rand(2, 10) * 100).to(dtype),
                (torch.rand(2, 10) * 100).to(dtype),
            ),
            tester_factory,
        )

    def test_multiply_f32_bcast_first(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 5, 1, 5),
            ),
            tester_factory,
        )

    def test_multiply_f32_bcast_second(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(4, 4, 2, 7),
                torch.randn(2, 7),
            ),
            tester_factory,
        )

    def test_multiply_f32_bcast_unary(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 1, 5),
            ),
            tester_factory,
        )
