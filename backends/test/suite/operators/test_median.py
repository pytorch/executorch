# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import torch
import unittest
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class MedianModel(torch.nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.median(x, dim=self.dim, keepdim=self.keepdim)


class MedianValueOnlyModel(torch.nn.Module):
    """Model that returns only the median values (not indices) when dim is specified."""

    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is not None:
            return torch.median(x, dim=self.dim, keepdim=self.keepdim)[0]
        else:
            return torch.median(x)


@operator_test
class Median(OperatorTest):
    @dtype_test
    def test_median_dtype(self, flow: TestFlow, dtype) -> None:
        # Test with different dtypes (global reduction)
        model = MedianValueOnlyModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype),), flow)

    def test_median_basic(self, flow: TestFlow) -> None:
        # Basic test with default parameters (global reduction)
        self._test_op(MedianValueOnlyModel(), (torch.randn(10, 10),), flow)

    def test_median_dim(self, flow: TestFlow) -> None:
        # Test with different dimensions (values only)

        # 2D tensor, dim=0
        self._test_op(MedianValueOnlyModel(dim=0), (torch.randn(5, 10),), flow)

        # 2D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(5, 10),), flow)

        # 3D tensor, dim=0
        self._test_op(MedianValueOnlyModel(dim=0), (torch.randn(3, 4, 5),), flow)

        # 3D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(3, 4, 5),), flow)

        # 3D tensor, dim=2
        self._test_op(MedianValueOnlyModel(dim=2), (torch.randn(3, 4, 5),), flow)

        # 4D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(2, 3, 4, 5),), flow)

        # Negative dim (last dimension)
        self._test_op(MedianValueOnlyModel(dim=-1), (torch.randn(3, 4, 5),), flow)

        # Negative dim (second-to-last dimension)
        self._test_op(MedianValueOnlyModel(dim=-2), (torch.randn(3, 4, 5),), flow)

    def test_median_with_indices(self, flow: TestFlow) -> None:
        # Test with different dimensions (values and indices)

        # 2D tensor, dim=0
        self._test_op(MedianModel(dim=0), (torch.randn(5, 10),), flow)

        # 2D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(5, 10),), flow)

        # 3D tensor, dim=0
        self._test_op(MedianModel(dim=0), (torch.randn(3, 4, 5),), flow)

        # 3D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(3, 4, 5),), flow)

        # 3D tensor, dim=2
        self._test_op(MedianModel(dim=2), (torch.randn(3, 4, 5),), flow)

        # 4D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(2, 3, 4, 5),), flow)

        # Negative dim (last dimension)
        self._test_op(MedianModel(dim=-1), (torch.randn(3, 4, 5),), flow)

        # Negative dim (second-to-last dimension)
        self._test_op(MedianModel(dim=-2), (torch.randn(3, 4, 5),), flow)

    def test_median_keepdim(self, flow: TestFlow) -> None:
        # Test with keepdim=True (values only)

        # 2D tensor, dim=0, keepdim=True
        self._test_op(
            MedianValueOnlyModel(dim=0, keepdim=True), (torch.randn(5, 10),), flow
        )

        # 2D tensor, dim=1, keepdim=True
        self._test_op(
            MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(5, 10),), flow
        )

        # 3D tensor, dim=1, keepdim=True
        self._test_op(
            MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), flow
        )

        # 4D tensor, dim=2, keepdim=True
        self._test_op(
            MedianValueOnlyModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), flow
        )

    def test_median_keepdim_with_indices(self, flow: TestFlow) -> None:
        # Test with keepdim=True (values and indices)

        # 2D tensor, dim=0, keepdim=True
        self._test_op(MedianModel(dim=0, keepdim=True), (torch.randn(5, 10),), flow)

        # 2D tensor, dim=1, keepdim=True
        self._test_op(MedianModel(dim=1, keepdim=True), (torch.randn(5, 10),), flow)

        # 3D tensor, dim=1, keepdim=True
        self._test_op(MedianModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), flow)

        # 4D tensor, dim=2, keepdim=True
        self._test_op(
            MedianModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), flow
        )

    def test_median_shapes(self, flow: TestFlow) -> None:
        # Test with different tensor shapes (global reduction)

        # 1D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(20),), flow)

        # 2D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(5, 10),), flow)

        # 3D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(3, 4, 5),), flow)

        # 4D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(2, 3, 4, 5),), flow)

        # 5D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(2, 2, 3, 4, 5),), flow)

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_median_edge_cases(self, flow: TestFlow) -> None:
        # Tensor with NaN (NaN should be propagated)
        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        self._test_op(
            MedianValueOnlyModel(), (x,), flow, generate_random_test_inputs=False
        )
        self._test_op(
            MedianValueOnlyModel(dim=0), (x,), flow, generate_random_test_inputs=False
        )
        self._test_op(
            MedianValueOnlyModel(dim=1), (x,), flow, generate_random_test_inputs=False
        )

    def test_median_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(MedianValueOnlyModel(), (torch.tensor([5.0]),), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (torch.tensor([5.0]),), flow)
