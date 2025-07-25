# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional, Tuple

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class MedianModel(torch.nn.Module):
    def __init__(
        self, 
        dim: Optional[int] = None, 
        keepdim: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        return torch.median(x, dim=self.dim, keepdim=self.keepdim)

class MedianValueOnlyModel(torch.nn.Module):
    """Model that returns only the median values (not indices) when dim is specified."""
    def __init__(
        self, 
        dim: Optional[int] = None, 
        keepdim: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        if self.dim is not None:
            return torch.median(x, dim=self.dim, keepdim=self.keepdim)[0]
        else:
            return torch.median(x)

@operator_test
class TestMedian(OperatorTest):
    @dtype_test
    def test_median_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes (global reduction)
        model = MedianValueOnlyModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype),), tester_factory)
        
    def test_median_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (global reduction)
        self._test_op(MedianValueOnlyModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_median_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions (values only)
        
        # 2D tensor, dim=0
        self._test_op(MedianValueOnlyModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(MedianValueOnlyModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(MedianValueOnlyModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(MedianValueOnlyModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(MedianValueOnlyModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(MedianValueOnlyModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_median_with_indices(self, tester_factory: Callable) -> None:
        # Test with different dimensions (values and indices)
        
        # 2D tensor, dim=0
        self._test_op(MedianModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(MedianModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(MedianModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(MedianModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(MedianModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(MedianModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_median_keepdim(self, tester_factory: Callable) -> None:
        # Test with keepdim=True (values only)
        
        # 2D tensor, dim=0, keepdim=True
        self._test_op(MedianValueOnlyModel(dim=0, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, keepdim=True
        self._test_op(MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, keepdim=True
        self._test_op(MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, keepdim=True
        self._test_op(MedianValueOnlyModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_median_keepdim_with_indices(self, tester_factory: Callable) -> None:
        # Test with keepdim=True (values and indices)
        
        # 2D tensor, dim=0, keepdim=True
        self._test_op(MedianModel(dim=0, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, keepdim=True
        self._test_op(MedianModel(dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, keepdim=True
        self._test_op(MedianModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, keepdim=True
        self._test_op(MedianModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_median_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes (global reduction)
        
        # 1D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(MedianValueOnlyModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_median_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns (global reduction)
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with odd number of elements (clear median)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with even number of elements (median is average of middle two)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with duplicate values
        x = torch.tensor([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        
    def test_median_dim_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns (dimension reduction)
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Tensor with odd number of elements in dimension
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        
        # Tensor with even number of elements in dimension
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Tensor with unsorted values
        x = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
    def test_median_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Tensor with NaN (NaN should be propagated)
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=1), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(MedianValueOnlyModel(), (x,), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (x,), tester_factory)
        
    def test_median_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(MedianValueOnlyModel(), (torch.tensor([5.0]),), tester_factory)
        self._test_op(MedianValueOnlyModel(dim=0), (torch.tensor([5.0]),), tester_factory)
