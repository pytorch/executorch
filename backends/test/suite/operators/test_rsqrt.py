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

    def test_rsqrt_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(RsqrtModel(), (torch.rand(20) + 0.01,), flow)

        # 2D tensor
        self._test_op(RsqrtModel(), (torch.rand(5, 10) + 0.01,), flow)

        # 3D tensor
        self._test_op(RsqrtModel(), (torch.rand(3, 4, 5) + 0.01,), flow)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_rsqrt_edge_cases(self, flow: TestFlow) -> None:
        # Tensor with infinity
        x = torch.tensor([float("inf"), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), flow, generate_random_test_inputs=False)
