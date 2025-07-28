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


class Model(torch.nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.elu(x, alpha=self.alpha, inplace=self.inplace)


@operator_test
class TestELU(OperatorTest):
    @dtype_test
    def test_elu_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 100).to(dtype),), flow)

    def test_elu_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_elu_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    def test_elu_f32_alpha(self, flow: TestFlow) -> None:
        self._test_op(Model(alpha=0.5), (torch.randn(3, 4, 5),), flow)

    def test_elu_f32_inplace(self, flow: TestFlow) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), flow)
