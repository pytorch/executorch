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
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.leaky_relu(
            x, negative_slope=self.negative_slope, inplace=self.inplace
        )


@operator_test
class TestLeakyReLU(OperatorTest):
    @dtype_test
    def test_leaky_relu_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 2 - 1).to(dtype),), flow)

    def test_leaky_relu_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_leaky_relu_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    def test_leaky_relu_f32_custom_slope(self, flow: TestFlow) -> None:
        self._test_op(Model(negative_slope=0.1), (torch.randn(3, 4, 5),), flow)

    def test_leaky_relu_f32_inplace(self, flow: TestFlow) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), flow)

    def test_leaky_relu_f32_boundary_values(self, flow: TestFlow) -> None:
        # Test with specific positive and negative values
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(Model(), (x,), flow)
