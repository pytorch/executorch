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
    def __init__(self, threshold=0.0, value=0.0, inplace=False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.threshold(
            x, threshold=self.threshold, value=self.value, inplace=self.inplace
        )


@operator_test
class TestThreshold(OperatorTest):
    @dtype_test
    def test_threshold_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(
            Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),), tester_factory
        )

    def test_threshold_f32_single_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(20),), tester_factory)

    def test_threshold_f32_multi_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), tester_factory)

    def test_threshold_f32_custom_threshold(self, tester_factory: Callable) -> None:
        self._test_op(Model(threshold=1.0), (torch.randn(3, 4, 5),), tester_factory)

    def test_threshold_f32_custom_value(self, tester_factory: Callable) -> None:
        self._test_op(Model(value=2.0), (torch.randn(3, 4, 5),), tester_factory)

    def test_threshold_f32_custom_threshold_value(
        self, tester_factory: Callable
    ) -> None:
        self._test_op(
            Model(threshold=0.5, value=1.0), (torch.randn(3, 4, 5),), tester_factory
        )

    def test_threshold_f32_inplace(self, tester_factory: Callable) -> None:
        self._test_op(Model(inplace=True), (torch.randn(3, 4, 5),), tester_factory)

    def test_threshold_f32_boundary_values(self, tester_factory: Callable) -> None:
        # Test with specific values around the threshold
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(Model(), (x,), tester_factory)

    def test_threshold_f32_all_params(self, tester_factory: Callable) -> None:
        # Test with all parameters customized
        self._test_op(
            Model(threshold=0.5, value=3.0, inplace=True),
            (torch.randn(3, 4, 5),),
            tester_factory,
        )
