# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class DiagonalModel(torch.nn.Module):
    def __init__(
        self,
        offset: int = 0,
        dim1: int = 0,
        dim2: int = 1
    ):
        super().__init__()
        self.offset = offset
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x):
        return torch.diagonal(x, offset=self.offset, dim1=self.dim1, dim2=self.dim2)

@operator_test
class TestDiagonal(OperatorTest):
    @dtype_test
    def test_diagonal_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = DiagonalModel().to(dtype)
        self._test_op(model, (torch.rand(5, 5).to(dtype),), tester_factory)
        
    def test_diagonal_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (main diagonal, dim1=0, dim2=1)
        
        # Square matrix
        self._test_op(DiagonalModel(), (torch.randn(5, 5),), tester_factory)
        
        # Rectangular matrix (tall)
        self._test_op(DiagonalModel(), (torch.randn(7, 5),), tester_factory)
        
        # Rectangular matrix (wide)
        self._test_op(DiagonalModel(), (torch.randn(5, 7),), tester_factory)
        
        # 3D tensor
        self._test_op(DiagonalModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(DiagonalModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_diagonal_offset(self, tester_factory: Callable) -> None:
        # Test with different offsets
        
        # Square matrix
        x = torch.randn(5, 5)
        
        # Main diagonal (offset=0)
        self._test_op(DiagonalModel(offset=0), (x,), tester_factory)
        
        # Super-diagonal (offset=1)
        self._test_op(DiagonalModel(offset=1), (x,), tester_factory)
        
        # Super-diagonal (offset=2)
        self._test_op(DiagonalModel(offset=2), (x,), tester_factory)
        
        # Sub-diagonal (offset=-1)
        self._test_op(DiagonalModel(offset=-1), (x,), tester_factory)
        
        # Sub-diagonal (offset=-2)
        self._test_op(DiagonalModel(offset=-2), (x,), tester_factory)
        
        # Rectangular matrix (tall)
        x = torch.randn(7, 5)
        
        # Main diagonal (offset=0)
        self._test_op(DiagonalModel(offset=0), (x,), tester_factory)
        
        # Super-diagonal (offset=1)
        self._test_op(DiagonalModel(offset=1), (x,), tester_factory)
        
        # Sub-diagonal (offset=-1)
        self._test_op(DiagonalModel(offset=-1), (x,), tester_factory)
        
        # Rectangular matrix (wide)
        x = torch.randn(5, 7)
        
        # Main diagonal (offset=0)
        self._test_op(DiagonalModel(offset=0), (x,), tester_factory)
        
        # Super-diagonal (offset=1)
        self._test_op(DiagonalModel(offset=1), (x,), tester_factory)
        
        # Sub-diagonal (offset=-1)
        self._test_op(DiagonalModel(offset=-1), (x,), tester_factory)
        
    def test_diagonal_dimensions(self, tester_factory: Callable) -> None:
        # Test with different dimension combinations
        
        # 3D tensor
        x = torch.randn(3, 4, 5)
        
        # Default: dim1=0, dim2=1
        self._test_op(DiagonalModel(dim1=0, dim2=1), (x,), tester_factory)
        
        # dim1=0, dim2=2
        self._test_op(DiagonalModel(dim1=0, dim2=2), (x,), tester_factory)
        
        # dim1=1, dim2=2
        self._test_op(DiagonalModel(dim1=1, dim2=2), (x,), tester_factory)
        
        # 4D tensor
        x = torch.randn(2, 3, 4, 5)
        
        # dim1=0, dim2=1
        self._test_op(DiagonalModel(dim1=0, dim2=1), (x,), tester_factory)
        
        # dim1=0, dim2=2
        self._test_op(DiagonalModel(dim1=0, dim2=2), (x,), tester_factory)
        
        # dim1=0, dim2=3
        self._test_op(DiagonalModel(dim1=0, dim2=3), (x,), tester_factory)
        
        # dim1=1, dim2=2
        self._test_op(DiagonalModel(dim1=1, dim2=2), (x,), tester_factory)
        
        # dim1=1, dim2=3
        self._test_op(DiagonalModel(dim1=1, dim2=3), (x,), tester_factory)
        
        # dim1=2, dim2=3
        self._test_op(DiagonalModel(dim1=2, dim2=3), (x,), tester_factory)
        
    def test_diagonal_negative_dimensions(self, tester_factory: Callable) -> None:
        # Test with negative dimensions
        
        # 3D tensor
        x = torch.randn(3, 4, 5)
        
        # dim1=-3, dim2=-2 (equivalent to dim1=0, dim2=1)
        self._test_op(DiagonalModel(dim1=-3, dim2=-2), (x,), tester_factory)
        
        # dim1=-3, dim2=-1 (equivalent to dim1=0, dim2=2)
        self._test_op(DiagonalModel(dim1=-3, dim2=-1), (x,), tester_factory)
        
        # dim1=-2, dim2=-1 (equivalent to dim1=1, dim2=2)
        self._test_op(DiagonalModel(dim1=-2, dim2=-1), (x,), tester_factory)
        
        # 4D tensor
        x = torch.randn(2, 3, 4, 5)
        
        # dim1=-4, dim2=-3 (equivalent to dim1=0, dim2=1)
        self._test_op(DiagonalModel(dim1=-4, dim2=-3), (x,), tester_factory)
        
        # dim1=-4, dim2=-1 (equivalent to dim1=0, dim2=3)
        self._test_op(DiagonalModel(dim1=-4, dim2=-1), (x,), tester_factory)
        
        # dim1=-2, dim2=-1 (equivalent to dim1=2, dim2=3)
        self._test_op(DiagonalModel(dim1=-2, dim2=-1), (x,), tester_factory)
        
    def test_diagonal_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 2D tensors
        self._test_op(DiagonalModel(), (torch.randn(3, 3),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(3, 5),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(5, 3),), tester_factory)
        
        # 3D tensors
        self._test_op(DiagonalModel(), (torch.randn(2, 3, 3),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 3, 5),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 5, 3),), tester_factory)
        
        # 4D tensors
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 3, 3),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 3, 5),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 5, 3),), tester_factory)
        
        # 5D tensors
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 2, 3, 3),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 2, 3, 5),), tester_factory)
        self._test_op(DiagonalModel(), (torch.randn(2, 2, 2, 5, 3),), tester_factory)
        
    def test_diagonal_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Identity matrix
        x = torch.eye(5, 5)
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]])
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Integer tensor
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
    def test_diagonal_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 3)
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 3)
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
        # Offset larger than matrix dimension (returns empty tensor)
        x = torch.randn(3, 3)
        self._test_op(DiagonalModel(offset=3), (x,), tester_factory)
        self._test_op(DiagonalModel(offset=-3), (x,), tester_factory)
        
        # 1x1 matrix
        x = torch.randn(1, 1)
        self._test_op(DiagonalModel(), (x,), tester_factory)
        
    def test_diagonal_combinations(self, tester_factory: Callable) -> None:
        # Test combinations of parameters
        
        # 3D tensor with different offsets and dimensions
        x = torch.randn(3, 4, 5)
        
        # Offset=1, dim1=0, dim2=1
        self._test_op(DiagonalModel(offset=1, dim1=0, dim2=1), (x,), tester_factory)
        
        # Offset=-1, dim1=0, dim2=2
        self._test_op(DiagonalModel(offset=-1, dim1=0, dim2=2), (x,), tester_factory)
        
        # Offset=2, dim1=1, dim2=2
        self._test_op(DiagonalModel(offset=2, dim1=1, dim2=2), (x,), tester_factory)
        
        # 4D tensor with different offsets and dimensions
        x = torch.randn(2, 3, 4, 5)
        
        # Offset=1, dim1=0, dim2=3
        self._test_op(DiagonalModel(offset=1, dim1=0, dim2=3), (x,), tester_factory)
        
        # Offset=-1, dim1=1, dim2=2
        self._test_op(DiagonalModel(offset=-1, dim1=1, dim2=2), (x,), tester_factory)
        
        # Offset=2, dim1=2, dim2=3
        self._test_op(DiagonalModel(offset=2, dim1=2, dim2=3), (x,), tester_factory)
