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


class UnsqueezeModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


@operator_test
class Unsqueeze(OperatorTest):
    @dtype_test
    def test_unsqueeze_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            UnsqueezeModel(dim=1),
            (torch.rand(3, 5).to(dtype),),
            flow,
        )

    def test_unsqueeze_basic(self, flow: TestFlow) -> None:
        self._test_op(
            UnsqueezeModel(dim=1),
            (torch.randn(3, 5),),
            flow,
        )

    def test_unsqueeze_positions(self, flow: TestFlow) -> None:
        self._test_op(
            UnsqueezeModel(dim=0),
            (torch.randn(3, 5),),
            flow,
        )

        self._test_op(
            UnsqueezeModel(dim=1),
            (torch.randn(3, 5),),
            flow,
        )

        self._test_op(
            UnsqueezeModel(dim=2),
            (torch.randn(3, 5),),
            flow,
        )

    def test_unsqueeze_negative_dim(self, flow: TestFlow) -> None:
        self._test_op(
            UnsqueezeModel(dim=-1),
            (torch.randn(3, 5),),
            flow,
        )

        self._test_op(
            UnsqueezeModel(dim=-2),
            (torch.randn(3, 5),),
            flow,
        )

        self._test_op(
            UnsqueezeModel(dim=-3),
            (torch.randn(3, 5),),
            flow,
        )

    def test_unsqueeze_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            UnsqueezeModel(dim=0),
            (torch.randn(5),),
            flow,
        )
        self._test_op(
            UnsqueezeModel(dim=1),
            (torch.randn(5),),
            flow,
        )

        self._test_op(
            UnsqueezeModel(dim=0),
            (torch.randn(3, 4, 5),),
            flow,
        )
        self._test_op(
            UnsqueezeModel(dim=2),
            (torch.randn(3, 4, 5),),
            flow,
        )
        self._test_op(
            UnsqueezeModel(dim=3),
            (torch.randn(3, 4, 5),),
            flow,
        )
