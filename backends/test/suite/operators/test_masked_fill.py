# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Union

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class MaskedFillModel(torch.nn.Module):
    def __init__(self, value: Union[float, int]):
        super().__init__()
        self.value = value

    def forward(self, x, mask):
        return x.masked_fill(mask, self.value)


@operator_test
class MaskedFill(OperatorTest):
    @dtype_test
    def test_masked_fill_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.rand(3, 4).to(dtype),
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                    ]
                ),
            ),
            flow,
        )

    def test_masked_fill_basic(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(3, 4),
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                    ]
                ),
            ),
            flow,
        )

    def test_masked_fill_different_values(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=5.0),
            (
                torch.randn(3, 4),
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                    ]
                ),
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=-5.0),
            (
                torch.randn(3, 4),
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                    ]
                ),
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=1),
            (
                torch.randn(3, 4),
                torch.tensor(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                    ]
                ),
            ),
            flow,
        )

    def test_masked_fill_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(5),
                torch.tensor([True, False, True, False, True]),
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(2, 3, 4),
                torch.tensor(
                    [
                        [
                            [True, False, True, False],
                            [False, True, False, True],
                            [True, True, False, False],
                        ],
                        [
                            [False, False, True, True],
                            [True, False, True, False],
                            [False, True, False, True],
                        ],
                    ]
                ),
            ),
            flow,
        )

    def test_masked_fill_all_true(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(3, 4),
                torch.ones(3, 4, dtype=torch.bool),
            ),
            flow,
        )

    def test_masked_fill_all_false(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(3, 4),
                torch.zeros(3, 4, dtype=torch.bool),
            ),
            flow,
        )

    def test_masked_fill_broadcast(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(3, 4),
                torch.tensor([True, False, True, False]),
            ),
            flow,
        )
