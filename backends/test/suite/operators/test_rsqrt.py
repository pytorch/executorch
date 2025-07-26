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


class RsqrtModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(x)


@operator_test
class TestRsqrt(OperatorTest):
    @dtype_test
    def test_rsqrt_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = RsqrtModel().to(dtype)
        # Use positive values only for rsqrt to avoid division by zero
        self._test_op(model, (torch.rand(10, 10).to(dtype) + 0.01,), flow)

    def test_rsqrt_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with positive values
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 10 + 0.01,), flow)

    def test_rsqrt_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(RsqrtModel(), (torch.rand(20) + 0.01,), flow)

        # 2D tensor
        self._test_op(RsqrtModel(), (torch.rand(5, 10) + 0.01,), flow)

        # 3D tensor
        self._test_op(RsqrtModel(), (torch.rand(3, 4, 5) + 0.01,), flow)

        # 4D tensor
        self._test_op(RsqrtModel(), (torch.rand(2, 3, 4, 5) + 0.01,), flow)

        # 5D tensor
        self._test_op(RsqrtModel(), (torch.rand(2, 2, 3, 4, 5) + 0.01,), flow)

    def test_rsqrt_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small values (rsqrt of small values gives large results)
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 0.01 + 0.01,), flow)
        # Perfect squares
        x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

    def test_rsqrt_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Tensor with specific values
        x = torch.tensor([1.0, 2.0, 4.0, 0.25, 0.5, 0.01])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with infinity
        x = torch.tensor([float("inf"), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Values very close to zero (but not zero)
        x = torch.tensor([1e-5, 1e-10, 1e-15])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Values where rsqrt(x) = 1/sqrt(x) has a simple result
        x = torch.tensor([1.0, 4.0, 9.0, 16.0])  # rsqrt gives [1.0, 0.5, 0.33..., 0.25]
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

    def test_rsqrt_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(
            RsqrtModel(),
            (torch.tensor([1.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            RsqrtModel(),
            (torch.tensor([4.0]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            RsqrtModel(),
            (torch.tensor([0.25]),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            RsqrtModel(),
            (torch.tensor([100.0]),),
            flow,
            generate_random_test_inputs=False,
        )
