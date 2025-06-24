# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class EqTensorModel(torch.nn.Module):
    """Model that compares two tensors for equality."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.eq(x, y)

class EqScalarModel(torch.nn.Module):
    """Model that compares a tensor with a scalar value."""
    def __init__(self, scalar: Union[int, float, bool]):
        super().__init__()
        self.scalar = scalar
        
    def forward(self, x):
        return torch.eq(x, self.scalar)

@operator_test
class TestEq(OperatorTest):
    @dtype_test
    def test_eq_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype == torch.bool:
            # For boolean tensors, use 0 and 1 values
            x = torch.randint(0, 2, (5, 5)).to(dtype)
            y = torch.randint(0, 2, (5, 5)).to(dtype)
        else:
            x = torch.rand(5, 5).to(dtype)
            y = torch.rand(5, 5).to(dtype)
            
        model = EqTensorModel().to(dtype)
        self._test_op(model, (x, y), tester_factory)
        
    def test_eq_tensor_basic(self, tester_factory: Callable) -> None:
        # Basic test comparing two tensors
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Test with identical tensors (should return all True)
        y = x.clone()
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
    def test_eq_scalar_basic(self, tester_factory: Callable) -> None:
        # Basic test comparing tensor with scalar
        x = torch.randn(5, 5)
        
        # Compare with scalar 0.0
        self._test_op(EqScalarModel(0.0), (x,), tester_factory)
        
        # Compare with scalar 1.0
        self._test_op(EqScalarModel(1.0), (x,), tester_factory)
        
        # Compare with scalar -1.0
        self._test_op(EqScalarModel(-1.0), (x,), tester_factory)
        
    def test_eq_broadcasting(self, tester_factory: Callable) -> None:
        # Test broadcasting with tensors of different shapes
        
        # Broadcasting 1D to 2D
        x = torch.randn(5, 5)
        y = torch.randn(5)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Broadcasting scalar tensor to 2D
        x = torch.randn(5, 5)
        y = torch.tensor([1.0])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Broadcasting with different dimensions
        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 1)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
    def test_eq_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensors
        x = torch.randn(10)
        y = torch.randn(10)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # 2D tensors
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # 3D tensors
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # 4D tensors
        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(2, 3, 4, 5)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # 5D tensors
        x = torch.randn(2, 2, 3, 4, 5)
        y = torch.randn(2, 2, 3, 4, 5)
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
    def test_eq_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Tensors with identical values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Tensors with some equal values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Tensors with no equal values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Integer tensors
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y = torch.tensor([[1, 0, 3], [0, 5, 0]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Boolean tensors
        x = torch.tensor([[True, False, True], [False, True, False]])
        y = torch.tensor([[True, False, False], [False, False, True]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
    def test_eq_scalar_values(self, tester_factory: Callable) -> None:
        # Test comparing tensors with different scalar values
        
        # Float tensor with float scalar
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(EqScalarModel(3.0), (x,), tester_factory)
        
        # Integer tensor with integer scalar
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_op(EqScalarModel(3), (x,), tester_factory)
        
        # Boolean tensor with boolean scalar
        x = torch.tensor([[True, False, True], [False, True, False]])
        self._test_op(EqScalarModel(True), (x,), tester_factory)
        
        # Float tensor with integer scalar
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._test_op(EqScalarModel(3), (x,), tester_factory)
        
        # Integer tensor with float scalar
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_op(EqScalarModel(3.0), (x,), tester_factory)
        
    def test_eq_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensors with NaN (NaN != NaN)
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        y = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Tensors with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        y = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Tensors with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        y = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Comparing infinity with infinity
        x = torch.tensor([float('inf')])
        y = torch.tensor([float('inf')])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Comparing -infinity with -infinity
        x = torch.tensor([float('-inf')])
        y = torch.tensor([float('-inf')])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Comparing infinity with -infinity
        x = torch.tensor([float('inf')])
        y = torch.tensor([float('-inf')])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
        # Single element tensors
        x = torch.tensor([5.0])
        y = torch.tensor([5.0])
        self._test_op(EqTensorModel(), (x, y), tester_factory)
        
    def test_eq_scalar_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases with scalar comparison
        
        # Comparing with NaN
        x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        self._test_op(EqScalarModel(float('nan')), (x,), tester_factory)
        
        # Comparing with infinity
        x = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, float('inf')]])
        self._test_op(EqScalarModel(float('inf')), (x,), tester_factory)
        
        # Comparing with negative infinity
        x = torch.tensor([[1.0, float('-inf'), 3.0], [4.0, 5.0, float('-inf')]])
        self._test_op(EqScalarModel(float('-inf')), (x,), tester_factory)
