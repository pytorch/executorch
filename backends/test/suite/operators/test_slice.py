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


class SliceSimple(torch.nn.Module):
    def __init__(self, index=1):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class SliceRange(torch.nn.Module):
    def forward(self, x):
        return x[1:3]


@operator_test
class Slice(OperatorTest):
    @dtype_test
    def test_slice_simple_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            SliceSimple().to(dtype),
            (torch.rand(2, 3, 4).to(dtype),),
            flow,
        )

    def test_slice_range(self, flow: TestFlow) -> None:
        self._test_op(
            SliceRange(),
            (torch.rand(2, 5, 4),),
            flow,
        )
