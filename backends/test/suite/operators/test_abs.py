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


class AbsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


@operator_test
class TestAbs(OperatorTest):
    @dtype_test
    def test_abs_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = AbsModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), flow)

    def test_abs_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with positive and negative values
        self._test_op(AbsModel(), (torch.randn(10, 10),), flow)

    def test_abs_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(AbsModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(AbsModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(AbsModel(), (torch.randn(3, 4, 5),), flow)

        # 4D tensor
        self._test_op(AbsModel(), (torch.randn(2, 3, 4, 5),), flow)

        # 5D tensor
        self._test_op(AbsModel(), (torch.randn(2, 2, 3, 4, 5),), flow)

    def test_abs_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 0.01,), flow)

        # Large values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 1000,), flow)

        # Mixed positive and negative values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 10,), flow)

        # All positive values
        self._test_op(AbsModel(), (torch.rand(10, 10) * 10,), flow)

        # All negative values
        self._test_op(AbsModel(), (torch.rand(10, 10) * -10,), flow)

        # Values close to zero
        self._test_op(AbsModel(), (torch.randn(10, 10) * 1e-5,), flow)

    def test_abs_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Zero tensor
        self._test_op(
            AbsModel(), (torch.zeros(10, 10),), flow, generate_random_test_inputs=False
        )

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(AbsModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(AbsModel(), (x,), flow, generate_random_test_inputs=False)

    def test_abs_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            AbsModel(), (torch.tensor([-5.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            AbsModel(), (torch.tensor([5.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            AbsModel(), (torch.tensor([0.0]),), flow, generate_random_test_inputs=False
        )
