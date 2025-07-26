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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.floor(x)


@operator_test
class TestFloor(OperatorTest):
    @dtype_test
    def test_floor_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = FloorModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), flow)

    def test_floor_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with fractional values
        self._test_op(FloorModel(), (torch.randn(10, 10),), flow)

    def test_floor_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(FloorModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(FloorModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(FloorModel(), (torch.randn(3, 4, 5),), flow)

        # 4D tensor
        self._test_op(FloorModel(), (torch.randn(2, 3, 4, 5),), flow)

        # 5D tensor
        self._test_op(FloorModel(), (torch.randn(2, 2, 3, 4, 5),), flow)

    def test_floor_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small fractional values
        self._test_op(FloorModel(), (torch.rand(10, 10) * 0.01,), flow)

        # Large fractional values
        self._test_op(FloorModel(), (torch.randn(10, 10) * 1000,), flow)

        # Mixed positive and negative values
        self._test_op(FloorModel(), (torch.randn(10, 10) * 10,), flow)

        # Values with specific fractional parts
        self._test_op(
            FloorModel(),
            (torch.arange(0, 10, 0.5).reshape(4, 5),),
            flow,
            generate_random_test_inputs=False,
        )

        # Values close to integers
        x = torch.randn(10, 10)
        x = x.round() + torch.rand(10, 10) * 0.01
        self._test_op(FloorModel(), (x,), flow)

    def test_floor_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Integer values
        self._test_op(
            FloorModel(),
            (torch.arange(10).reshape(2, 5).float(),),
            flow,
            generate_random_test_inputs=False,
        )

        # Zero tensor
        self._test_op(
            FloorModel(),
            (torch.zeros(10, 10),),
            flow,
            generate_random_test_inputs=False,
        )

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)

        # Values just below integers
        x = torch.arange(10).float() - 0.01
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)

        # Values just above integers
        x = torch.arange(10).float() + 0.01
        self._test_op(FloorModel(), (x,), flow, generate_random_test_inputs=False)

    def test_floor_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            FloorModel(),
            (torch.tensor([1.5]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            FloorModel(),
            (torch.tensor([-1.5]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            FloorModel(),
            (torch.tensor([0.0]),),
            flow,
            generate_random_test_inputs=False,
        )
