# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class CumsumModel(torch.nn.Module):
    def __init__(
        self, 
        dim: int,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        
    def forward(self, x):
        return torch.cumsum(x, dim=self.dim, dtype=self.dtype)

@operator_test
class TestCumsum(OperatorTest):
    @dtype_test
    def test_cumsum_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = CumsumModel(dim=0).to(dtype)
        self._test_op(model, (torch.rand(5, 5).to(dtype),), tester_factory)
        
    def test_cumsum_basic(self, tester_factory: Callable) -> None:
        # Basic test with different dimensions
        
        # 1D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(10),), tester_factory)
        
        # 2D tensor, dim=0
        self._test_op(CumsumModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(CumsumModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(CumsumModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(CumsumModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(CumsumModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_cumsum_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions
        
        # 2D tensor, dim=-1 (last dimension)
        self._test_op(CumsumModel(dim=-1), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=-2 (second-to-last dimension)
        self._test_op(CumsumModel(dim=-2), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=-1 (last dimension)
        self._test_op(CumsumModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=-2 (second-to-last dimension)
        self._test_op(CumsumModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=-3 (third-to-last dimension)
        self._test_op(CumsumModel(dim=-3), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_cumsum_output_dtype(self, tester_factory: Callable) -> None:
        # Test with explicit output dtype
        
        # Float input with float32 output
        self._test_op(CumsumModel(dim=0, dtype=torch.float32), (torch.randn(5, 10),), tester_factory)
        
        # Float input with float64 output
        self._test_op(CumsumModel(dim=0, dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        
        # Integer input with int64 output
        self._test_op(CumsumModel(dim=0, dtype=torch.int64), (torch.randint(0, 10, (5, 10)),), tester_factory)
        
        # Integer input with float output
        self._test_op(CumsumModel(dim=0, dtype=torch.float32), (torch.randint(0, 10, (5, 10)),), tester_factory)
        
    def test_cumsum_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(5, 10),), tester_factory)
        self._test_op(CumsumModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(2, 3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=2), (torch.randn(2, 3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=3), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(CumsumModel(dim=0), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        self._test_op(CumsumModel(dim=4), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_cumsum_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensor with sequential values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Integer tensor
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
    def test_cumsum_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        
    def test_cumsum_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(CumsumModel(dim=0), (torch.tensor([5.0]),), tester_factory)
        
    def test_cumsum_large_values(self, tester_factory: Callable) -> None:
        # Test with large values that might cause overflow
        x = torch.tensor([[1e10, 1e10, 1e10], [1e10, 1e10, 1e10]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
        
        # Test with large integer values
        x = torch.tensor([[1000000, 1000000, 1000000], [1000000, 1000000, 1000000]])
        self._test_op(CumsumModel(dim=0), (x,), tester_factory)
        self._test_op(CumsumModel(dim=1), (x,), tester_factory)
