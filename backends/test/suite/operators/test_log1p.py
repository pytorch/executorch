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


class Log1pModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log1p(x)


@operator_test
class TestLog1p(OperatorTest):
    @dtype_test
    def test_log1p_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = Log1pModel().to(dtype)
        # Use values greater than -1 for log1p
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 0.5,), flow)

    def test_log1p_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with values greater than -1
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 2 - 0.5,), flow)

    def test_log1p_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(Log1pModel(), (torch.rand(20) * 2 - 0.5,), flow)

        # 2D tensor
        self._test_op(Log1pModel(), (torch.rand(5, 10) * 2 - 0.5,), flow)

        # 3D tensor
        self._test_op(Log1pModel(), (torch.rand(3, 4, 5) * 2 - 0.5,), flow)

        # 4D tensor
        self._test_op(Log1pModel(), (torch.rand(2, 3, 4, 5) * 2 - 0.5,), flow)

        # 5D tensor
        self._test_op(Log1pModel(), (torch.rand(2, 2, 3, 4, 5) * 2 - 0.5,), flow)

    def test_log1p_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small values close to zero
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.01,), flow)

        # Values close to -1 (lower bound for log1p)
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.1 - 0.99,), flow)

        # Values around 0 (log1p(0) = 0)
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.2 - 0.1,), flow)

        # Medium positive values
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 10,), flow)

        # Large positive values
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 1000,), flow)

        # Very large positive values
        self._test_op(Log1pModel(), (torch.rand(5, 5) * 1e10,), flow)

    def test_log1p_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Zero tensor (log1p(0) = 0)
        self._test_op(
            Log1pModel(),
            (torch.zeros(10, 10),),
            flow,
            generate_random_test_inputs=False,
        )

        # Tensor with specific values
        x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

        # Values very close to -1
        x = torch.tensor([-0.999, -0.9999, -0.99999])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

        # Very small positive values (where log1p is more accurate than log(1+x))
        x = torch.tensor([1e-10, 1e-15, 1e-20])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

    def test_log1p_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            Log1pModel(),
            (torch.tensor([0.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Log1pModel(),
            (torch.tensor([1.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Log1pModel(),
            (torch.tensor([-0.5]),),
            flow,
            generate_random_test_inputs=False,
        )
