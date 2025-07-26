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


class ExpandModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.expand(self.shape)


@operator_test
class Expand(OperatorTest):
    @dtype_test
    def test_expand_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            ExpandModel(shape=[3, 5]),
            (torch.rand(1, 5).to(dtype),),
            flow,
        )

    def test_expand_basic(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[3, 5]),
            (torch.randn(1, 5),),
            flow,
        )

    def test_expand_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[3, 5]),
            (torch.randn(1, 5),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[3, 4]),
            (torch.randn(1, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[2, 1, 5]),
            (torch.randn(1, 5),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[3, 2, 5]),
            (torch.randn(3, 1, 5),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[3, 5, 2]),
            (torch.randn(3, 5, 1),),
            flow,
        )

    def test_expand_keep_original_size(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[3, -1]),
            (torch.randn(1, 5),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[-1, 5]),
            (torch.randn(2, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[-1, 4, -1]),
            (torch.randn(2, 1, 3),),
            flow,
        )

    def test_expand_singleton_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[5]),
            (torch.randn(1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[3, 4]),
            (torch.randn(1, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[3, 5]),
            (torch.randn(5),),
            flow,
        )
