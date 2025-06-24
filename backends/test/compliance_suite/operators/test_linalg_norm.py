# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Optional, Tuple, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class LinalgNormModel(torch.nn.Module):
    def __init__(
        self,
        ord: Optional[Union[int, float, str]] = None,
        dim: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        
    def forward(self, x):
        return torch.linalg.norm(x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype)

@operator_test
class TestLinalgNorm(OperatorTest):
    @dtype_test
    def test_linalg_norm_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype.is_complex:
            # Skip complex dtypes for now
            return
        
        model = LinalgNormModel().to(dtype)
        self._test_op(model, (torch.rand(5, 5).to(dtype),), tester_factory)
        
    def test_linalg_norm_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (Frobenius norm for matrices, L2 norm for vectors)
        
        # Vector norm (1D tensor)
        self._test_op(LinalgNormModel(), (torch.randn(10),), tester_factory)
        
        # Matrix norm (2D tensor)
        self._test_op(LinalgNormModel(), (torch.randn(5, 10),), tester_factory)
        
        # Higher-dimensional tensor
        self._test_op(LinalgNormModel(), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_linalg_norm_vector_ord(self, tester_factory: Callable) -> None:
        # Test vector norms with different ord values
        
        # 1D tensor
        x = torch.randn(10)
        
        # L1 norm (sum of absolute values)
        self._test_op(LinalgNormModel(ord=1), (x,), tester_factory)
        
        # L2 norm (Euclidean norm, default)
        self._test_op(LinalgNormModel(ord=2), (x,), tester_factory)
        
        # L-infinity norm (maximum absolute value)
        self._test_op(LinalgNormModel(ord=float('inf')), (x,), tester_factory)
        
        # L-negative-infinity norm (minimum absolute value)
        self._test_op(LinalgNormModel(ord=float('-inf')), (x,), tester_factory)
        
        # L0 "norm" (number of non-zero elements)
        self._test_op(LinalgNormModel(ord=0), (x,), tester_factory)
        
        # Arbitrary p-norm
        self._test_op(LinalgNormModel(ord=3.5), (x,), tester_factory)
        
    def test_linalg_norm_matrix_ord(self, tester_factory: Callable) -> None:
        # Test matrix norms with different ord values
        
        # 2D tensor
        x = torch.randn(5, 10)
        
        # Frobenius norm (default)
        self._test_op(LinalgNormModel(ord='fro'), (x,), tester_factory)
        
        # Nuclear norm (sum of singular values)
        self._test_op(LinalgNormModel(ord='nuc'), (x,), tester_factory)
        
        # 1-norm (maximum absolute column sum)
        self._test_op(LinalgNormModel(ord=1), (x,), tester_factory)
        
        # 2-norm (largest singular value)
        self._test_op(LinalgNormModel(ord=2), (x,), tester_factory)
        
        # Infinity norm (maximum absolute row sum)
        self._test_op(LinalgNormModel(ord=float('inf')), (x,), tester_factory)
        
        # -1 norm (minimum absolute column sum)
        self._test_op(LinalgNormModel(ord=-1), (x,), tester_factory)
        
        # -2 norm (smallest singular value)
        self._test_op(LinalgNormModel(ord=-2), (x,), tester_factory)
        
        # -infinity norm (minimum absolute row sum)
        self._test_op(LinalgNormModel(ord=float('-inf')), (x,), tester_factory)
        
    def test_linalg_norm_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions
        
        # 2D tensor, dim=0 (column vector norms)
        self._test_op(LinalgNormModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1 (row vector norms)
        self._test_op(LinalgNormModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(LinalgNormModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(LinalgNormModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(LinalgNormModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(0,1) (matrix norm for each depth slice)
        self._test_op(LinalgNormModel(dim=(0, 1)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(0,2) (matrix norm for each height slice)
        self._test_op(LinalgNormModel(dim=(0, 2)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(1,2) (matrix norm for each batch)
        self._test_op(LinalgNormModel(dim=(1, 2)), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_linalg_norm_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions
        
        # 2D tensor, dim=-1 (last dimension)
        self._test_op(LinalgNormModel(dim=-1), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=-2 (second-to-last dimension)
        self._test_op(LinalgNormModel(dim=-2), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=(-1,-2) (matrix norm for each batch)
        self._test_op(LinalgNormModel(dim=(-1, -2)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(-1,-3) (matrix norm for each height slice)
        self._test_op(LinalgNormModel(dim=(-1, -3)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(-2,-3) (matrix norm for each depth slice)
        self._test_op(LinalgNormModel(dim=(-2, -3)), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_linalg_norm_keepdim(self, tester_factory: Callable) -> None:
        # Test with keepdim=True
        
        # Vector norm with keepdim
        self._test_op(LinalgNormModel(keepdim=True), (torch.randn(10),), tester_factory)
        
        # Matrix norm with keepdim
        self._test_op(LinalgNormModel(keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # Vector norm along specific dimension with keepdim
        self._test_op(LinalgNormModel(dim=0, keepdim=True), (torch.randn(5, 10),), tester_factory)
        self._test_op(LinalgNormModel(dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # Matrix norm along specific dimensions with keepdim
        self._test_op(LinalgNormModel(dim=(0, 1), keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_linalg_norm_output_dtype(self, tester_factory: Callable) -> None:
        # Test with explicit output dtype
        
        # Float input with float32 output
        self._test_op(LinalgNormModel(dtype=torch.float32), (torch.randn(5, 10),), tester_factory)
        
        # Float input with float64 output
        self._test_op(LinalgNormModel(dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        
        # Integer input with float32 output
        self._test_op(LinalgNormModel(dtype=torch.float32), (torch.randint(0, 10, (5, 10)),), tester_factory)
        
        # Integer input with float64 output
        self._test_op(LinalgNormModel(dtype=torch.float64), (torch.randint(0, 10, (5, 10)),), tester_factory)
        
    def test_linalg_norm_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor (vector)
        self._test_op(LinalgNormModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor (matrix)
        self._test_op(LinalgNormModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(LinalgNormModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(LinalgNormModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(LinalgNormModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_linalg_norm_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Integer tensor
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
    def test_linalg_norm_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(LinalgNormModel(), (x,), tester_factory)
        
    def test_linalg_norm_combinations(self, tester_factory: Callable) -> None:
        # Test combinations of parameters
        
        # Vector norm with specific ord and dim
        self._test_op(LinalgNormModel(ord=1, dim=1), (torch.randn(5, 10),), tester_factory)
        self._test_op(LinalgNormModel(ord=2, dim=0), (torch.randn(5, 10),), tester_factory)
        self._test_op(LinalgNormModel(ord=float('inf'), dim=1), (torch.randn(5, 10),), tester_factory)
        
        # Matrix norm with specific ord and dim
        self._test_op(LinalgNormModel(ord='fro', dim=(0, 1)), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(LinalgNormModel(ord=1, dim=(1, 2)), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(LinalgNormModel(ord=2, dim=(0, 2)), (torch.randn(3, 4, 5),), tester_factory)
        
        # Vector norm with specific ord, dim, and keepdim
        self._test_op(LinalgNormModel(ord=1, dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # Matrix norm with specific ord, dim, and keepdim
        self._test_op(LinalgNormModel(ord='fro', dim=(0, 1), keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # Vector norm with specific ord, dim, and dtype
        self._test_op(LinalgNormModel(ord=1, dim=1, dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        
        # Matrix norm with specific ord, dim, and dtype
        self._test_op(LinalgNormModel(ord='fro', dim=(0, 1), dtype=torch.float64), (torch.randn(3, 4, 5),), tester_factory)
        
        # Full combination: ord, dim, keepdim, and dtype
        self._test_op(LinalgNormModel(ord=2, dim=1, keepdim=True, dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        self._test_op(LinalgNormModel(ord='fro', dim=(0, 1), keepdim=True, dtype=torch.float64), (torch.randn(3, 4, 5),), tester_factory)
