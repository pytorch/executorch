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


class Log10Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log10(x)


@operator_test
class TestLog10(OperatorTest):
    @dtype_test
    def test_log10_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = Log10Model().to(dtype)
        # Use positive values only for log10
        self._test_op(model, (torch.rand(10, 10).to(dtype) + 0.01,), flow)

    def test_log10_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with positive values
        self._test_op(Log10Model(), (torch.rand(10, 10) + 0.01,), flow)

    def test_log10_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(Log10Model(), (torch.rand(20) + 0.01,), flow)

        # 2D tensor
        self._test_op(Log10Model(), (torch.rand(5, 10) + 0.01,), flow)

        # 3D tensor
        self._test_op(Log10Model(), (torch.rand(3, 4, 5) + 0.01,), flow)

        # 4D tensor
        self._test_op(Log10Model(), (torch.rand(2, 3, 4, 5) + 0.01,), flow)

        # 5D tensor
        self._test_op(Log10Model(), (torch.rand(2, 2, 3, 4, 5) + 0.01,), flow)

    def test_log10_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small positive values (close to zero)
        self._test_op(Log10Model(), (torch.rand(10, 10) * 0.01 + 1e-6,), flow)

        # Values around 1 (log10(1) = 0)
        self._test_op(Log10Model(), (torch.rand(10, 10) * 0.2 + 0.9,), flow)

        # Values around powers of 10
        self._test_op(
            Log10Model(),
            (torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0]).reshape(5, 1),),
            flow,
            generate_random_test_inputs=False,
        )

        # Medium values
        self._test_op(Log10Model(), (torch.rand(10, 10) * 10 + 0.01,), flow)

        # Large values
        self._test_op(Log10Model(), (torch.rand(10, 10) * 1000 + 0.01,), flow)

        # Very large values
        self._test_op(Log10Model(), (torch.rand(5, 5) * 1e10 + 0.01,), flow)

    def test_log10_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Values very close to zero
        self._test_op(Log10Model(), (torch.rand(10, 10) * 1e-6 + 1e-10,), flow)

        # Tensor with specific values
        x = torch.tensor([1.0, 10.0, 100.0, 1000.0, 0.1, 0.01])
        self._test_op(Log10Model(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), 1.0, 10.0])
        self._test_op(Log10Model(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, 10.0])
        self._test_op(Log10Model(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with ones (log10(1) = 0)
        self._test_op(
            Log10Model(), (torch.ones(10, 10),), flow, generate_random_test_inputs=False
        )

        # Tensor with tens (log10(10) = 1)
        self._test_op(
            Log10Model(),
            (torch.ones(10, 10) * 10,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_log10_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            Log10Model(),
            (torch.tensor([1.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Log10Model(),
            (torch.tensor([10.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Log10Model(),
            (torch.tensor([0.1]),),
            flow,
            generate_random_test_inputs=False,
        )
