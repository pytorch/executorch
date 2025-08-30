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


class FloorModel(torch.nn.Module):
    def forward(self, x):
        return torch.floor(x)


@operator_test
class TestFloor(OperatorTest):
    @dtype_test
    def test_floor_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = FloorModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), flow)

    def test_floor_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(FloorModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(FloorModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(FloorModel(), (torch.randn(3, 4, 5),), flow)

    def test_floor_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)
