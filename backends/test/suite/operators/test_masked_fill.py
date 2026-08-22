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
        mask = torch.randint(0, 2, (16, 32), dtype=torch.bool)
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.rand(16, 32).to(dtype),
                mask,
            ),
            flow,
        )

    def test_masked_fill_different_values(self, flow: TestFlow) -> None:
        mask = torch.randint(0, 2, (16, 32), dtype=torch.bool)

        self._test_op(
            MaskedFillModel(value=5.0),
            (
                torch.randn(16, 32),
                mask,
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=-5.0),
            (
                torch.randn(16, 32),
                mask,
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=1),
            (
                torch.randn(16, 32),
                mask,
            ),
            flow,
        )

    def test_masked_fill_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(512),
                torch.randint(0, 2, (512,), dtype=torch.bool),
            ),
            flow,
        )

        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(4, 8, 16),
                torch.randint(0, 2, (4, 8, 16), dtype=torch.bool),
            ),
            flow,
        )

    def test_masked_fill_broadcast(self, flow: TestFlow) -> None:
        self._test_op(
            MaskedFillModel(value=0.0),
            (
                torch.randn(16, 32),
                torch.randint(0, 2, (32,), dtype=torch.bool),
            ),
            flow,
        )
