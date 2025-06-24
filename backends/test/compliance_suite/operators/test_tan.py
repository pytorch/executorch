# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class TanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.tan(x)

@operator_test
class TestTan(OperatorTest):
    @dtype_test
    def test_tan_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = TanModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_tan_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(TanModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_tan_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(TanModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(TanModel(), (torch.randn(5, 5),), tester_factory)
        
        # 3D tensor
        self._test_op(TanModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(TanModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(TanModel(), (torch.randn(2, 2, 2, 2, 2),), tester_factory)
        
    def test_tan_specific_values(self, tester_factory: Callable) -> None:
        # Test with specific values
        # tan has singularities at (n+1/2)*pi
        self._test_op(
            TanModel(), 
            (torch.tensor([0.0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 2.0, -2.0]),), 
            tester_factory
        )
        
    def test_tan_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero values
        self._test_op(TanModel(), (torch.zeros(5),), tester_factory)
        
        # Values near zero
        self._test_op(
            TanModel(), 
            (torch.tensor([1e-5, -1e-5, 1e-10, -1e-10]),), 
            tester_factory
        )
        
        # Values away from singularities
        self._test_op(
            TanModel(), 
            (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),), 
            tester_factory
        )
        
        # Infinity and NaN
        self._test_op(
            TanModel(), 
            (torch.tensor([float('inf'), float('-inf'), float('nan')]),), 
            tester_factory
        )
        
        # Single element tensor
        self._test_op(TanModel(), (torch.tensor([0.5]),), tester_factory)
