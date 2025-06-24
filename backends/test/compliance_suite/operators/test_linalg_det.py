# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class LinalgDetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.linalg.det(x)

@operator_test
class TestLinalgDet(OperatorTest):
    @dtype_test
    def test_linalg_det_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype.is_complex:
            # Skip complex dtypes for now
            return
        
        model = LinalgDetModel().to(dtype)
        # Create a square matrix
        x = torch.rand(3, 3).to(dtype)
        self._test_op(model, (x,), tester_factory)
        
    def test_linalg_det_basic(self, tester_factory: Callable) -> None:
        # Basic test with square matrices
        
        # 2x2 matrix
        self._test_op(LinalgDetModel(), (torch.randn(2, 2),), tester_factory)
        
        # 3x3 matrix
        self._test_op(LinalgDetModel(), (torch.randn(3, 3),), tester_factory)
        
        # 4x4 matrix
        self._test_op(LinalgDetModel(), (torch.randn(4, 4),), tester_factory)
        
        # 5x5 matrix
        self._test_op(LinalgDetModel(), (torch.randn(5, 5),), tester_factory)
        
    def test_linalg_det_batch(self, tester_factory: Callable) -> None:
        # Test with batches of matrices
        
        # Batch of 2x2 matrices
        self._test_op(LinalgDetModel(), (torch.randn(5, 2, 2),), tester_factory)
        
        # Batch of 3x3 matrices
        self._test_op(LinalgDetModel(), (torch.randn(5, 3, 3),), tester_factory)
        
        # Multiple batches of 2x2 matrices
        self._test_op(LinalgDetModel(), (torch.randn(4, 5, 2, 2),), tester_factory)
        
        # Multiple batches of 3x3 matrices
        self._test_op(LinalgDetModel(), (torch.randn(4, 5, 3, 3),), tester_factory)
        
    def test_linalg_det_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 2D tensors (single matrix)
        self._test_op(LinalgDetModel(), (torch.randn(2, 2),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(3, 3),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(4, 4),), tester_factory)
        
        # 3D tensors (batch of matrices)
        self._test_op(LinalgDetModel(), (torch.randn(2, 2, 2),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(2, 3, 3),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(5, 4, 4),), tester_factory)
        
        # 4D tensors (multiple batches)
        self._test_op(LinalgDetModel(), (torch.randn(2, 3, 2, 2),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(2, 3, 3, 3),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(2, 5, 4, 4),), tester_factory)
        
        # 5D tensors (multiple batch dimensions)
        self._test_op(LinalgDetModel(), (torch.randn(2, 2, 3, 2, 2),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(2, 2, 3, 3, 3),), tester_factory)
        self._test_op(LinalgDetModel(), (torch.randn(2, 2, 5, 4, 4),), tester_factory)
        
    def test_linalg_det_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Identity matrix (determinant should be 1)
        x = torch.eye(3, 3)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Diagonal matrix
        x = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Upper triangular matrix
        x = torch.triu(torch.randn(3, 3))
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Lower triangular matrix
        x = torch.tril(torch.randn(3, 3))
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with negative values
        x = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with mixed positive and negative values
        x = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Integer matrix
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
    def test_linalg_det_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Matrix with all same values (determinant should be 0)
        x = torch.ones(3, 3)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Zero matrix (determinant should be 0)
        x = torch.zeros(3, 3)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with a zero row (determinant should be 0)
        x = torch.randn(3, 3)
        x[1] = torch.zeros(3)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with a zero column (determinant should be 0)
        x = torch.randn(3, 3)
        x[:, 1] = torch.zeros(3)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with duplicate rows (determinant should be 0)
        x = torch.randn(3, 3)
        x[1] = x[0]
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Matrix with duplicate columns (determinant should be 0)
        x = torch.randn(3, 3)
        x[:, 1] = x[:, 0]
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # 1x1 matrix
        x = torch.randn(1, 1)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
    def test_linalg_det_special_matrices(self, tester_factory: Callable) -> None:
        # Test with special matrices
        
        # Orthogonal matrix (determinant should be +1 or -1)
        # Create a simple rotation matrix (which is orthogonal)
        theta = torch.tensor(0.5)
        c, s = torch.cos(theta), torch.sin(theta)
        x = torch.tensor([[c, -s], [s, c]])
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Symmetric matrix
        x = torch.randn(3, 3)
        x = x + x.t()  # Make symmetric
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Skew-symmetric matrix (determinant of odd-sized skew-symmetric matrix is 0)
        x = torch.randn(3, 3)
        x = x - x.t()  # Make skew-symmetric
        x.fill_diagonal_(0)  # Diagonal must be zero for skew-symmetric
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Permutation matrix
        x = torch.eye(3)
        # Swap rows to create a permutation matrix
        x = x[[1, 0, 2]]
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Singular matrix (determinant should be 0)
        # Create a matrix with linearly dependent rows
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]])  # Third row is sum of first two
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
    def test_linalg_det_batch_mixed(self, tester_factory: Callable) -> None:
        # Test with batches containing different types of matrices
        
        # Batch with identity, zero, and random matrices
        x = torch.randn(3, 3, 3)
        x[0] = torch.eye(3)  # First matrix is identity
        x[1] = torch.zeros(3, 3)  # Second matrix is zero
        # Third matrix is random (already set)
        self._test_op(LinalgDetModel(), (x,), tester_factory)
        
        # Batch with singular and non-singular matrices
        x = torch.randn(2, 3, 3)
        # Make the first matrix singular
        x[0, 0] = x[0, 1] + x[0, 2]  # First row is sum of other rows
        self._test_op(LinalgDetModel(), (x,), tester_factory)
