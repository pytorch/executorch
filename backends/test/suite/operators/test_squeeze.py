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


class SqueezeModel(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


class SqueezeDimModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


@operator_test
class Squeeze(OperatorTest):
    @dtype_test
    def test_squeeze_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            SqueezeModel(),
            (torch.rand(1, 3, 1, 5).to(dtype),),
            flow,
        )

    def test_squeeze_specific_dimension(self, flow: TestFlow) -> None:
        self._test_op(
            SqueezeDimModel(dim=0),
            (torch.randn(1, 3, 5),),
            flow,
        )

        self._test_op(
            SqueezeDimModel(dim=2),
            (torch.randn(3, 4, 1, 5),),
            flow,
        )

        self._test_op(
            SqueezeDimModel(dim=-1),
            (torch.randn(3, 4, 5, 1),),
            flow,
        )

    def test_squeeze_no_effect(self, flow: TestFlow) -> None:
        self._test_op(
            SqueezeDimModel(dim=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            SqueezeModel(),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_squeeze_multiple_dims(self, flow: TestFlow) -> None:
        self._test_op(
            SqueezeModel(),
            (torch.randn(1, 3, 1, 5, 1),),
            flow,
        )

        self._test_op(
            SqueezeDimModel(dim=(0, 1)),
            (torch.randn(1, 1, 1),),
            flow,
        )
