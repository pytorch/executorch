# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SortModel(torch.nn.Module):
    def __init__(
        self, 
        dim: int = -1, 
        descending: bool = False,
        stable: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.descending = descending
        self.stable = stable
        
    def forward(self, x):
        return torch.sort(x, dim=self.dim, descending=self.descending, stable=self.stable)

class SortValuesOnlyModel(torch.nn.Module):
    """Model that returns only the sorted values (not indices)."""
    def __init__(
        self, 
        dim: int = -1, 
        descending: bool = False,
        stable: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.descending = descending
        self.stable = stable
        
    def forward(self, x):
        return torch.sort(x, dim=self.dim, descending=self.descending, stable=self.stable)[0]

@operator_test
class TestSort(OperatorTest):
    @dtype_test
    def test_sort_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SortValuesOnlyModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype),), tester_factory)
        
    def test_sort_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(SortValuesOnlyModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_sort_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions (values only)
        
        # 2D tensor, dim=0
        self._test_op(SortValuesOnlyModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(SortValuesOnlyModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(SortValuesOnlyModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(SortValuesOnlyModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(SortValuesOnlyModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(SortValuesOnlyModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(SortValuesOnlyModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(SortValuesOnlyModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_sort_with_indices(self, tester_factory: Callable) -> None:
        # Test with different dimensions (values and indices)
        
        # 2D tensor, dim=0
        self._test_op(SortModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(SortModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(SortModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(SortModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(SortModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(SortModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(SortModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(SortModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_sort_descending(self, tester_factory: Callable) -> None:
        # Test with descending=True (values only)
        
        # 2D tensor, dim=0, descending=True
        self._test_op(SortValuesOnlyModel(dim=0, descending=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, descending=True
        self._test_op(SortValuesOnlyModel(dim=1, descending=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, descending=True
        self._test_op(SortValuesOnlyModel(dim=1, descending=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, descending=True
        self._test_op(SortValuesOnlyModel(dim=2, descending=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_sort_descending_with_indices(self, tester_factory: Callable) -> None:
        # Test with descending=True (values and indices)
        
        # 2D tensor, dim=0, descending=True
        self._test_op(SortModel(dim=0, descending=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, descending=True
        self._test_op(SortModel(dim=1, descending=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, descending=True
        self._test_op(SortModel(dim=1, descending=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, descending=True
        self._test_op(SortModel(dim=2, descending=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_sort_stable(self, tester_factory: Callable) -> None:
        # Test with stable=True (values only)
        
        # 2D tensor, dim=0, stable=True
        self._test_op(SortValuesOnlyModel(dim=0, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, stable=True
        self._test_op(SortValuesOnlyModel(dim=1, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, stable=True
        self._test_op(SortValuesOnlyModel(dim=1, stable=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, stable=True
        self._test_op(SortValuesOnlyModel(dim=2, stable=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_sort_stable_with_indices(self, tester_factory: Callable) -> None:
        # Test with stable=True (values and indices)
        
        # 2D tensor, dim=0, stable=True
        self._test_op(SortModel(dim=0, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, stable=True
        self._test_op(SortModel(dim=1, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, stable=True
        self._test_op(SortModel(dim=1, stable=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, stable=True
        self._test_op(SortModel(dim=2, stable=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_sort_descending_stable(self, tester_factory: Callable) -> None:
        # Test with descending=True and stable=True
        
        # 2D tensor, dim=0, descending=True, stable=True
        self._test_op(SortValuesOnlyModel(dim=0, descending=True, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, descending=True, stable=True
        self._test_op(SortValuesOnlyModel(dim=1, descending=True, stable=True), (torch.randn(5, 10),), tester_factory)
        
        # With indices
        self._test_op(SortModel(dim=0, descending=True, stable=True), (torch.randn(5, 10),), tester_factory)
        self._test_op(SortModel(dim=1, descending=True, stable=True), (torch.randn(5, 10),), tester_factory)
        
    def test_sort_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(SortValuesOnlyModel(), (torch.randn(20),), tester_factory)
        self._test_op(SortModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(SortValuesOnlyModel(), (torch.randn(5, 10),), tester_factory)
        self._test_op(SortModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(SortValuesOnlyModel(), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(SortModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(SortValuesOnlyModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        self._test_op(SortModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(SortValuesOnlyModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        self._test_op(SortModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_sort_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with unsorted values
        x = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with duplicate values
        x = torch.tensor([[3.0, 3.0, 2.0], [6.0, 6.0, 5.0]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
    def test_sort_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Tensor with NaN (NaN should be at the end for ascending sort)
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(SortValuesOnlyModel(), (x,), tester_factory)
        self._test_op(SortModel(), (x,), tester_factory)
        
    def test_sort_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(SortValuesOnlyModel(), (torch.tensor([5.0]),), tester_factory)
        self._test_op(SortModel(), (torch.tensor([5.0]),), tester_factory)
