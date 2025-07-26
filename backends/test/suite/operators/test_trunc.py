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


class TruncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.trunc(x)


@operator_test
class TestTrunc(OperatorTest):
    @dtype_test
    def test_trunc_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = TruncModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 10 - 5,), flow)

    def test_trunc_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 5,), flow)

    def test_trunc_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(TruncModel(), (torch.randn(20) * 5,), flow)

        # 2D tensor
        self._test_op(TruncModel(), (torch.randn(5, 10) * 5,), flow)

        # 3D tensor
        self._test_op(TruncModel(), (torch.randn(3, 4, 5) * 5,), flow)

        # 4D tensor
        self._test_op(TruncModel(), (torch.randn(2, 3, 4, 5) * 5,), flow)

        # 5D tensor
        self._test_op(TruncModel(), (torch.randn(2, 2, 3, 4, 5) * 5,), flow)

    def test_trunc_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 0.1,), flow)

        # Medium fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 5,), flow)

        # Large fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 1000,), flow)

        # Mixed positive and negative values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 10,), flow)

        # Values with specific fractional parts
        x = torch.arange(-5, 5, 0.5)  # [-5.0, -4.5, -4.0, ..., 4.0, 4.5]
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

    def test_trunc_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Integer values (should remain unchanged)
        self._test_op(
            TruncModel(),
            (torch.arange(-5, 6).float(),),
            flow,
            generate_random_test_inputs=False,
        )

        # Values with different fractional parts
        x = torch.tensor(
            [-2.9, -2.5, -2.1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 2.1, 2.5, 2.9]
        )
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

        # Zero tensor
        self._test_op(
            TruncModel(),
            (torch.zeros(10, 10),),
            flow,
            generate_random_test_inputs=False,
        )

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.4, -1.4])
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.4, -1.4])
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

        # Very large values (where fractional part becomes insignificant)
        x = torch.tensor([1e10, 1e10 + 0.4, 1e10 + 0.6])
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

        # Very small values close to zero
        x = torch.tensor([-0.1, -0.01, -0.001, 0.001, 0.01, 0.1])
        self._test_op(TruncModel(), (x,), flow, generate_random_test_inputs=False)

    def test_trunc_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            TruncModel(),
            (torch.tensor([1.4]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([1.5]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([1.6]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([-1.4]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([-1.5]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([-1.6]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            TruncModel(),
            (torch.tensor([0.0]),),
            flow,
            generate_random_test_inputs=False,
        )
