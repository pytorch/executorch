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


class ClampModel(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


@operator_test
class TestClamp(OperatorTest):
    @dtype_test
    def test_clamp_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes
        model = ClampModel(min_val=-0.5, max_val=0.5).to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), flow)

    def test_clamp_min_only(self, flow: TestFlow) -> None:
        # Test with only min value specified
        self._test_op(ClampModel(min_val=0.0), (torch.randn(10, 10),), flow)

    def test_clamp_max_only(self, flow: TestFlow) -> None:
        # Test with only max value specified
        self._test_op(ClampModel(max_val=0.0), (torch.randn(10, 10),), flow)

    def test_clamp_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes
        model = ClampModel(min_val=-1.0, max_val=1.0)

        # 1D tensor
        self._test_op(model, (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(model, (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(model, (torch.randn(3, 4, 5),), flow)

    def test_clamp_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Min equals max
        self._test_op(
            ClampModel(min_val=0.0, max_val=0.0), (torch.randn(10, 10),), flow
        )

        # Tensor with infinity
        x = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        self._test_op(
            ClampModel(min_val=-2.0, max_val=2.0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        # Tensor with NaN
        x = torch.tensor([float("nan"), 1.0, -1.0])
        self._test_op(
            ClampModel(min_val=-2.0, max_val=2.0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
