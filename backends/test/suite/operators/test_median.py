# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import torch
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

    def test_median_values(self, flow: TestFlow) -> None:
        # Test with different value patterns (global reduction)

        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with odd number of elements (clear median)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with even number of elements (median is average of middle two)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with duplicate values
        x = torch.tensor([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)

    def test_median_dim_values(self, flow: TestFlow) -> None:
        # Test with different value patterns (dimension reduction)

        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Tensor with odd number of elements in dimension
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)

        # Tensor with even number of elements in dimension
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Tensor with unsorted values
        x = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

    def test_median_edge_cases(self, flow: TestFlow) -> None:
        # Test edge cases

        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Tensor with infinity
        x = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("inf")]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Tensor with negative infinity
        x = torch.tensor([[1.0, float("-inf"), 3.0], [4.0, 5.0, float("-inf")]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Tensor with NaN (NaN should be propagated)
        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), flow)

        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(MedianValueOnlyModel(), (x,), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), flow)

    def test_median_scalar(self, flow: TestFlow) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(MedianValueOnlyModel(), (torch.tensor([5.0]),), flow)
        self._test_op(MedianValueOnlyModel(dim=0), (torch.tensor([5.0]),), flow)
