# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.tanh(x)

@operator_test
class TestTanh(OperatorTest):
    @dtype_test
    def test_tanh_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = TanhModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_tanh_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(TanhModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_tanh_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(TanhModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(TanhModel(), (torch.randn(5, 5),), tester_factory)
        
        # 3D tensor
        self._test_op(TanhModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(TanhModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(TanhModel(), (torch.randn(2, 2, 2, 2, 2),), tester_factory)
        
    def test_tanh_specific_values(self, tester_factory: Callable) -> None:
        # Test with specific values
        # tanh has range (-1, 1) and approaches ±1 as x approaches ±∞
        self._test_op(
            TanhModel(), 
            (torch.tensor([0.0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 2.0, -2.0, 5.0, -5.0]),), 
            tester_factory
        )
        
    def test_tanh_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero values
        self._test_op(TanhModel(), (torch.zeros(5),), tester_factory)
        
        # Values near zero
        self._test_op(
            TanhModel(), 
            (torch.tensor([1e-5, -1e-5, 1e-10, -1e-10]),), 
            tester_factory
        )
        
        # Large values (tanh approaches ±1)
        self._test_op(
            TanhModel(), 
            (torch.tensor([10.0, -10.0, 20.0, -20.0]),), 
            tester_factory
        )
        
        # Infinity and NaN
        self._test_op(
            TanhModel(), 
            (torch.tensor([float('inf'), float('-inf'), float('nan')]),), 
            tester_factory
        )
        
        # Single element tensor
        self._test_op(TanhModel(), (torch.tensor([0.5]),), tester_factory)
