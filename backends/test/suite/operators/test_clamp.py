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

    def test_clamp_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters
        # Input: tensor with values outside the clamp range
        self._test_op(
            ClampModel(min_val=-0.5, max_val=0.5), (torch.randn(10, 10),), flow
        )

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

        # 4D tensor
        self._test_op(model, (torch.randn(2, 3, 4, 5),), flow)

        # 5D tensor
        self._test_op(model, (torch.randn(2, 2, 3, 4, 5),), flow)

    def test_clamp_values(self, flow: TestFlow) -> None:
        # Test with different value ranges

        # Small values with narrow clamp range
        self._test_op(
            ClampModel(min_val=-0.01, max_val=0.01), (torch.randn(10, 10) * 0.1,), flow
        )

        # Large values with wide clamp range
        self._test_op(
            ClampModel(min_val=-100, max_val=100), (torch.randn(10, 10) * 1000,), flow
        )

        # Mixed positive and negative values
        self._test_op(
            ClampModel(min_val=-5, max_val=5), (torch.randn(10, 10) * 10,), flow
        )

        # All values within clamp range
        self._test_op(ClampModel(min_val=-10, max_val=10), (torch.randn(10, 10),), flow)

        # All values outside clamp range (below min)
        self._test_op(
            ClampModel(min_val=1.0, max_val=2.0), (torch.randn(10, 10) - 10,), flow
        )

        # All values outside clamp range (above max)
        self._test_op(
            ClampModel(min_val=-2.0, max_val=-1.0), (torch.randn(10, 10) + 10,), flow
        )

    def test_clamp_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Zero tensor
        self._test_op(
            ClampModel(min_val=-1.0, max_val=1.0),
            (torch.zeros(10, 10),),
            flow,
            generate_random_test_inputs=False,
        )

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

        # Values at exactly min/max bounds
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        self._test_op(
            ClampModel(min_val=-0.5, max_val=0.5),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_clamp_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        model = ClampModel(min_val=-1.0, max_val=1.0)
        self._test_op(
            model, (torch.tensor([-5.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            model, (torch.tensor([5.0]),), flow, generate_random_test_inputs=False
        )
        self._test_op(
            model, (torch.tensor([0.0]),), flow, generate_random_test_inputs=False
        )
