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


class Model(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.prelu(x)


@operator_test
class TestPReLU(OperatorTest):
    @dtype_test
    def test_prelu_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(Model().to(dtype), ((torch.rand(2, 10) * 2 - 1).to(dtype),), flow)

    def test_prelu_f32_single_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(20),), flow)

    def test_prelu_f32_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), flow)

    def test_prelu_f32_custom_init(self, flow: TestFlow) -> None:
        self._test_op(Model(init=0.1), (torch.randn(3, 4, 5),), flow)

    def test_prelu_f32_channel_shared(self, flow: TestFlow) -> None:
        # Default num_parameters=1 means the parameter is shared across all channels
        self._test_op(Model(num_parameters=1), (torch.randn(2, 3, 4, 5),), flow)

    def test_prelu_f32_per_channel_parameter(self, flow: TestFlow) -> None:
        # num_parameters=3 means each channel has its own parameter (for dim=1)
        self._test_op(Model(num_parameters=3), (torch.randn(2, 3, 4, 5),), flow)

    def test_prelu_f32_boundary_values(self, flow: TestFlow) -> None:
        # Test with specific positive and negative values
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(Model(), (x,), flow)
