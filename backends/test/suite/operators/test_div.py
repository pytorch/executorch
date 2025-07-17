# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Callable, Optional

import torch

from executorch.backends.test.suite import dtype_test, operator_test, OperatorTest


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x / y


class ModelWithRounding(torch.nn.Module):
    def __init__(self, rounding_mode: Optional[str]):
        super().__init__()
        self.rounding_mode = rounding_mode

    def forward(self, x, y):
        return torch.div(x, y, rounding_mode=self.rounding_mode)


@operator_test
class Divide(OperatorTest):
    @dtype_test
    def test_divide_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                (torch.rand(2, 10) * 100).to(dtype),
                (torch.rand(2, 10) * 100 + 0.1).to(
                    dtype
                ),  # Adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )

    def test_divide_f32_bcast_first(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 5, 1, 5).abs()
                + 0.1,  # Using abs and adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )

    def test_divide_f32_bcast_second(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(4, 4, 2, 7),
                torch.randn(2, 7).abs()
                + 0.1,  # Using abs and adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )

    def test_divide_f32_bcast_unary(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                torch.randn(5),
                torch.randn(1, 1, 5).abs()
                + 0.1,  # Using abs and adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )

    def test_divide_f32_trunc(self, tester_factory: Callable) -> None:
        self._test_op(
            ModelWithRounding(rounding_mode="trunc"),
            (
                torch.randn(3, 4) * 10,
                torch.randn(3, 4).abs()
                + 0.1,  # Using abs and adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )

    def test_divide_f32_floor(self, tester_factory: Callable) -> None:
        self._test_op(
            ModelWithRounding(rounding_mode="floor"),
            (
                torch.randn(3, 4) * 10,
                torch.randn(3, 4).abs()
                + 0.1,  # Using abs and adding 0.1 to avoid division by zero
            ),
            tester_factory,
        )
