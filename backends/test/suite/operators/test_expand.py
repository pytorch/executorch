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
            ExpandModel(shape=[8, 32]),
            (torch.rand(1, 32).to(dtype),),
            flow,
        )

    def test_expand_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[8, 32]),
            (torch.randn(1, 32),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[16, 20]),
            (torch.randn(1, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[4, 1, 32]),
            (torch.randn(1, 32),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[8, 4, 16]),
            (torch.randn(8, 1, 16),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[6, 16, 8]),
            (torch.randn(6, 16, 1),),
            flow,
        )

    def test_expand_keep_original_size(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[8, -1]),
            (torch.randn(1, 32),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[-1, 32]),
            (torch.randn(4, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[-1, 16, -1]),
            (torch.randn(4, 1, 8),),
            flow,
        )

    def test_expand_rank_increase(self, flow: TestFlow) -> None:
        # Test expanding 2D tensor to 3D
        self._test_op(
            ExpandModel(shape=[6, 8, 16]),
            (torch.randn(8, 16),),
            flow,
        )

        # Test expanding 2D tensor to 4D
        self._test_op(
            ExpandModel(shape=[3, 4, 8, 16]),
            (torch.randn(8, 16),),
            flow,
        )

    def test_expand_singleton_dimensions(self, flow: TestFlow) -> None:
        self._test_op(
            ExpandModel(shape=[512]),
            (torch.randn(1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[16, 20]),
            (torch.randn(1, 1),),
            flow,
        )

        self._test_op(
            ExpandModel(shape=[8, 32]),
            (torch.randn(32),),
            flow,
        )
