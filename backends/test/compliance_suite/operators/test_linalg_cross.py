# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class LinalgCrossModel(torch.nn.Module):
    def __init__(
        self,
        dim: int = -1
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, y):
        return torch.linalg.cross(x, y, dim=self.dim)

@operator_test
class TestLinalgCross(OperatorTest):
    @dtype_test
    def test_linalg_cross_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype.is_complex:
            # Skip complex dtypes for now
            return
        
        model = LinalgCrossModel().to(dtype)
        # Create two 3D vectors
        x = torch.rand(3).to(dtype)
        y = torch.rand(3).to(dtype)
        self._test_op(model, (x, y), tester_factory)
        
    def test_linalg_cross_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (dim=-1)
        
        # Simple 3D vectors
        x = torch.randn(3)
        y = torch.randn(3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Batch of 3D vectors
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Multiple batches of 3D vectors
        x = torch.randn(4, 5, 3)
        y = torch.randn(4, 5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
    def test_linalg_cross_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions
        
        # 1D tensor, dim=0
        x = torch.randn(3)
        y = torch.randn(3)
        self._test_op(LinalgCrossModel(dim=0), (x, y), tester_factory)
        
        # 2D tensor, dim=0
        x = torch.randn(3, 5)
        y = torch.randn(3, 5)
        self._test_op(LinalgCrossModel(dim=0), (x, y), tester_factory)
        
        # 2D tensor, dim=1
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        self._test_op(LinalgCrossModel(dim=1), (x, y), tester_factory)
        
        # 3D tensor, dim=0
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        self._test_op(LinalgCrossModel(dim=0), (x, y), tester_factory)
        
        # 3D tensor, dim=1
        x = torch.randn(4, 3, 5)
        y = torch.randn(4, 3, 5)
        self._test_op(LinalgCrossModel(dim=1), (x, y), tester_factory)
        
        # 3D tensor, dim=2
        x = torch.randn(4, 5, 3)
        y = torch.randn(4, 5, 3)
        self._test_op(LinalgCrossModel(dim=2), (x, y), tester_factory)
        
    def test_linalg_cross_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions
        
        # 1D tensor, dim=-1
        x = torch.randn(3)
        y = torch.randn(3)
        self._test_op(LinalgCrossModel(dim=-1), (x, y), tester_factory)
        
        # 2D tensor, dim=-1
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        self._test_op(LinalgCrossModel(dim=-1), (x, y), tester_factory)
        
        # 2D tensor, dim=-2
        x = torch.randn(3, 5)
        y = torch.randn(3, 5)
        self._test_op(LinalgCrossModel(dim=-2), (x, y), tester_factory)
        
        # 3D tensor, dim=-1
        x = torch.randn(4, 5, 3)
        y = torch.randn(4, 5, 3)
        self._test_op(LinalgCrossModel(dim=-1), (x, y), tester_factory)
        
        # 3D tensor, dim=-2
        x = torch.randn(4, 3, 5)
        y = torch.randn(4, 3, 5)
        self._test_op(LinalgCrossModel(dim=-2), (x, y), tester_factory)
        
        # 3D tensor, dim=-3
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        self._test_op(LinalgCrossModel(dim=-3), (x, y), tester_factory)
        
    def test_linalg_cross_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensors
        x = torch.randn(3)
        y = torch.randn(3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # 2D tensors
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # 3D tensors
        x = torch.randn(4, 5, 3)
        y = torch.randn(4, 5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # 4D tensors
        x = torch.randn(2, 4, 5, 3)
        y = torch.randn(2, 4, 5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # 5D tensors
        x = torch.randn(2, 2, 4, 5, 3)
        y = torch.randn(2, 2, 4, 5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
    def test_linalg_cross_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Standard basis vectors
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Orthogonal vectors
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 0.0, 1.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Parallel vectors (cross product should be zero)
        x = torch.tensor([2.0, 4.0, 6.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([-1.0, -2.0, -3.0])
        y = torch.tensor([-4.0, -5.0, -6.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([-1.0, 2.0, -3.0])
        y = torch.tensor([4.0, -5.0, 6.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([0.5, 1.5, 2.5])
        y = torch.tensor([3.5, 4.5, 5.5])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Integer tensor
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
    def test_linalg_cross_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero vectors
        x = torch.zeros(3)
        y = torch.zeros(3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # One zero vector
        x = torch.zeros(3)
        y = torch.randn(3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Unit vectors
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Batch with some zero vectors
        x = torch.randn(5, 3)
        x[2] = torch.zeros(3)
        y = torch.randn(5, 3)
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
        
        # Batch with some parallel vectors
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        y[1] = 2 * x[1]  # Make one pair of vectors parallel
        self._test_op(LinalgCrossModel(), (x, y), tester_factory)
