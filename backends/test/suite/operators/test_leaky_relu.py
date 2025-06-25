# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Callable

import torch

from executorch.backends.test.suite import dtype_test, operator_test, OperatorTest


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
    def test_leaky_relu_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 2 - 1).to(dtype),), tester_factory)

    def test_leaky_relu_f32_single_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(20),), tester_factory)

    def test_leaky_relu_f32_multi_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), tester_factory)

    def test_leaky_relu_f32_custom_slope(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(negative_slope=0.1), (torch.randn(3, 4, 5),), tester_factory
        )

    def test_leaky_relu_f32_inplace(self, tester_factory: Callable) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), tester_factory)

    def test_leaky_relu_f32_boundary_values(self, tester_factory: Callable) -> None:
        # Test with specific positive and negative values
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(Model(), (x,), tester_factory)
