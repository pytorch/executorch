# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ExpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.exp(x)

@operator_test
class TestExp(OperatorTest):
    @dtype_test
    def test_exp_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = ExpModel().to(dtype)
        # Use smaller range to avoid overflow
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 4 - 2,), tester_factory)
        
    def test_exp_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with values in a reasonable range
        self._test_op(ExpModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_exp_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(ExpModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(ExpModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(ExpModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(ExpModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(ExpModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_exp_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values (close to zero)
        self._test_op(ExpModel(), (torch.randn(10, 10) * 0.01,), tester_factory)
        
        # Medium values
        self._test_op(ExpModel(), (torch.randn(10, 10),), tester_factory)
        
        # Large negative values (exp approaches 0)
        self._test_op(ExpModel(), (torch.randn(10, 10) * -10,), tester_factory)
        
        # Large positive values (exp grows rapidly)
        # Use smaller tensor to avoid excessive memory usage
        self._test_op(ExpModel(), (torch.randn(5, 5) * 5,), tester_factory)
        
        # Values around 1 (exp(0) = 1)
        self._test_op(ExpModel(), (torch.randn(10, 10) * 0.1,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(ExpModel(), (torch.randn(10, 10) * 2,), tester_factory)
        
    def test_exp_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor (exp(0) = 1)
        self._test_op(ExpModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5])
        self._test_op(ExpModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0])
        self._test_op(ExpModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, -1.0])
        self._test_op(ExpModel(), (x,), tester_factory)
        
        # Very large negative values (should approach zero)
        x = torch.tensor([-100.0, -1000.0])
        self._test_op(ExpModel(), (x,), tester_factory)
        
    def test_exp_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(ExpModel(), (torch.tensor([0.0]),), tester_factory)
        self._test_op(ExpModel(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(ExpModel(), (torch.tensor([-1.0]),), tester_factory)
