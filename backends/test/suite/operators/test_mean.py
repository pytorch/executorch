# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Optional, Tuple, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class MeanModel(torch.nn.Module):
    def __init__(
        self, 
        dim: Optional[Union[int, Tuple[int, ...], List[int]]] = None, 
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        
    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype)

@operator_test
class TestMean(OperatorTest):
    @dtype_test
    def test_mean_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = MeanModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype),), tester_factory)
        
    def test_mean_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (global reduction)
        self._test_op(MeanModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_mean_dim(self, tester_factory: Callable) -> None:
        # Test with different dimensions
        
        # 2D tensor, dim=0
        self._test_op(MeanModel(dim=0), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1
        self._test_op(MeanModel(dim=1), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=0
        self._test_op(MeanModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=1
        self._test_op(MeanModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=2
        self._test_op(MeanModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=1
        self._test_op(MeanModel(dim=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Negative dim (last dimension)
        self._test_op(MeanModel(dim=-1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Negative dim (second-to-last dimension)
        self._test_op(MeanModel(dim=-2), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_mean_multi_dim(self, tester_factory: Callable) -> None:
        # Test with multiple dimensions
        
        # 3D tensor, dim=(0, 1)
        self._test_op(MeanModel(dim=(0, 1)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(0, 2)
        self._test_op(MeanModel(dim=(0, 2)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 3D tensor, dim=(1, 2)
        self._test_op(MeanModel(dim=(1, 2)), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=(1, 3)
        self._test_op(MeanModel(dim=(1, 3)), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=(0, 2)
        self._test_op(MeanModel(dim=(0, 2)), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=(-1, -3)
        self._test_op(MeanModel(dim=(-1, -3)), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 4D tensor, all dimensions
        self._test_op(MeanModel(dim=(0, 1, 2, 3)), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_mean_keepdim(self, tester_factory: Callable) -> None:
        # Test with keepdim=True
        
        # 2D tensor, dim=0, keepdim=True
        self._test_op(MeanModel(dim=0, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 2D tensor, dim=1, keepdim=True
        self._test_op(MeanModel(dim=1, keepdim=True), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor, dim=1, keepdim=True
        self._test_op(MeanModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor, dim=2, keepdim=True
        self._test_op(MeanModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Multiple dimensions with keepdim=True
        self._test_op(MeanModel(dim=(1, 2), keepdim=True), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_mean_output_dtype(self, tester_factory: Callable) -> None:
        # Test with explicit output dtype
        
        # Integer input with float output
        self._test_op(MeanModel(dtype=torch.float32), (torch.randint(0, 10, (5, 10)),), tester_factory)
        
        # Float input with specified float output
        self._test_op(MeanModel(dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        
        # With dimension reduction and dtype
        self._test_op(MeanModel(dim=1, dtype=torch.float64), (torch.randn(5, 10),), tester_factory)
        
    def test_mean_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(MeanModel(), (torch.randn(20),), tester_factory)
        self._test_op(MeanModel(dim=0), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(MeanModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(MeanModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(MeanModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(MeanModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_mean_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensor with integer sequence
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with duplicate values
        x = torch.tensor([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([[-3.0, 2.0, -1.0], [6.0, -5.0, 4.0]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
    def test_mean_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with all same values
        x = torch.ones(3, 4)
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Zero tensor
        x = torch.zeros(3, 4)
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Tensor with NaN (NaN should be propagated)
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
        
        # Single element tensor
        x = torch.tensor([5.0])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        
    def test_mean_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(MeanModel(), (torch.tensor([5.0]),), tester_factory)
        self._test_op(MeanModel(dim=0), (torch.tensor([5.0]),), tester_factory)
        
    def test_mean_integer_division(self, tester_factory: Callable) -> None:
        # Test with integer tensors (should produce float results)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_op(MeanModel(), (x,), tester_factory)
        self._test_op(MeanModel(dim=0), (x,), tester_factory)
        self._test_op(MeanModel(dim=1), (x,), tester_factory)
