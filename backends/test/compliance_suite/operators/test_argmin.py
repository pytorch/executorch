# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ArgminModel(torch.nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        return torch.argmin(x, dim=self.dim, keepdim=self.keepdim)

@operator_test
class TestArgmin(OperatorTest):
    @dtype_test
    def test_argmin_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = ArgminModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype),), tester_factory)
        
    def test_argmin_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (flattened tensor)
        self._test_op(ArgminModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_argmin_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions
        
        # 2D tensor, dim=0
        self._test_op(ArgminModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(ArgminModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(ArgminModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(ArgminModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(ArgminModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(ArgminModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(ArgminModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(ArgminModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_argmin_keepdim(self, tester_factory: Callable) -> None:
        # Test with keepdim=True
        
        # 2D tensor, dim=0, keepdim=True
        self._test_op(ArgminModel(dim=0, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, keepdim=True
        self._test_op(ArgminModel(dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, keepdim=True
        self._test_op(ArgminModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, keepdim=True
        self._test_op(ArgminModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_argmin_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(ArgminModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(ArgminModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(ArgminModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(ArgminModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(ArgminModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_argmin_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensor with clear minimum
        x = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Tensor with duplicate minimum values (should return first occurrence)
        x = torch.tensor([[3.0, 2.0, 2.0], [1.0, 1.0, 5.0]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
    def test_argmin_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Tensor with NaN (NaN should be ignored in comparison)
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(ArgminModel(), (x,), tester_factory)
        self._test_op(ArgminModel(dim=0), (x,), tester_factory)
        self._test_op(ArgminModel(dim=1), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(ArgminModel(), (x,), tester_factory)
        
    def test_argmin_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(ArgminModel(), (torch.tensor([5.0]),), tester_factory)
