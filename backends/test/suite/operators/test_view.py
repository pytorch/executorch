# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class ViewModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


@operator_test
class View(OperatorTest):
    @dtype_test
    def test_view_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            ViewModel(shape=[3, 5]),
            (torch.rand(15).to(dtype),),
            flow,
        )

    def test_view_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            ViewModel(shape=[3, 5]),
            (torch.randn(15),),
            flow,
        )

        self._test_op(
            ViewModel(shape=[20]),
            (torch.randn(4, 5),),
            flow,
        )

        self._test_op(
            ViewModel(shape=[2, 2, 5]),
            (torch.randn(4, 5),),
            flow,
        )

        self._test_op(
            ViewModel(shape=[6, 4]),
            (torch.randn(3, 2, 4),),
            flow,
        )

    def test_view_inferred_dimension(self, flow: TestFlow) -> None:
        self._test_op(
            ViewModel(shape=[3, -1]),
            (torch.randn(15),),
            flow,
        )

        self._test_op(
            ViewModel(shape=[-1, 5]),
            (torch.randn(15),),
            flow,
        )

        self._test_op(
            ViewModel(shape=[2, -1, 3]),
            (torch.randn(24),),
            flow,
        )
