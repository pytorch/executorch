# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class RoundModel(torch.nn.Module):
    def __init__(self, decimals=None):
        super().__init__()
        self.decimals = decimals

    def forward(self, x):
        if self.decimals is not None:
            return torch.round(x, decimals=self.decimals)
        return torch.round(x)


@operator_test
class TestRound(OperatorTest):
    @dtype_test
    def test_round_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = RoundModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 10 - 5,), flow)

    def test_round_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(RoundModel(), (torch.randn(20) * 5,), flow)

        # 2D tensor
        self._test_op(RoundModel(), (torch.randn(5, 10) * 5,), flow)

        # 3D tensor
        self._test_op(RoundModel(), (torch.randn(3, 4, 5) * 5,), flow)

    def test_round_values(self, flow: TestFlow) -> None:
        # Values with specific fractional parts
        x = torch.arange(-5, 5, 0.5)  # [-5.0, -4.5, -4.0, ..., 4.0, 4.5]
        self._test_op(RoundModel(), (x,), flow, generate_random_test_inputs=False)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_round_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Values exactly halfway between integers (should round to even)
        x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        self._test_op(RoundModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.4, -1.4])
        self._test_op(RoundModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.4, -1.4])
        self._test_op(RoundModel(), (x,), flow, generate_random_test_inputs=False)

        # Very large values (where fractional part becomes insignificant)
        x = torch.tensor([1e10, 1e10 + 0.4, 1e10 + 0.6])
        self._test_op(RoundModel(), (x,), flow, generate_random_test_inputs=False)

    def test_round_decimals(self, flow: TestFlow) -> None:
        # Test with different decimal places

        # Round to 1 decimal place
        x = torch.tensor([1.44, 1.45, 1.46, -1.44, -1.45, -1.46])
        self._test_op(
            RoundModel(decimals=1), (x,), flow, generate_random_test_inputs=False
        )

        # Round to 2 decimal places
        x = torch.tensor([1.444, 1.445, 1.446, -1.444, -1.445, -1.446])
        self._test_op(
            RoundModel(decimals=2), (x,), flow, generate_random_test_inputs=False
        )

        # Round to negative decimal places (tens)
        x = torch.tensor([14.4, 15.5, 16.6, -14.4, -15.5, -16.6])
        self._test_op(
            RoundModel(decimals=-1), (x,), flow, generate_random_test_inputs=False
        )

        # Round to negative decimal places (hundreds)
        x = torch.tensor([144.4, 155.5, 166.6, -144.4, -155.5, -166.6])
        self._test_op(
            RoundModel(decimals=-2), (x,), flow, generate_random_test_inputs=False
        )

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_round_decimals_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases with decimal places

        # Infinity and NaN with various decimal places
        x = torch.tensor([float("inf"), float("-inf"), float("nan")])
        self._test_op(
            RoundModel(decimals=2), (x,), flow, generate_random_test_inputs=False
        )
        self._test_op(
            RoundModel(decimals=-2), (x,), flow, generate_random_test_inputs=False
        )

        # Values exactly at the rounding threshold for different decimal places
        x = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self._test_op(
            RoundModel(decimals=1), (x,), flow, generate_random_test_inputs=False
        )

        # Negative values exactly at the rounding threshold
        x = torch.tensor(
            [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]
        )
        self._test_op(
            RoundModel(decimals=1), (x,), flow, generate_random_test_inputs=False
        )
