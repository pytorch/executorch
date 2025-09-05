# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import unittest
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

    def test_log1p_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes

        # 1D tensor
        self._test_op(Log1pModel(), (torch.rand(20) * 2 - 0.5,), flow)

        # 2D tensor
        self._test_op(Log1pModel(), (torch.rand(5, 10) * 2 - 0.5,), flow)

        # 3D tensor
        self._test_op(Log1pModel(), (torch.rand(3, 4, 5) * 2 - 0.5,), flow)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_log1p_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases
        # Tensor with infinity
        x = torch.tensor([float("inf"), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)

        # Tensor with NaN
        x = torch.tensor([float("nan"), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), flow, generate_random_test_inputs=False)
