# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class CoshModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.cosh(x)

@operator_test
class TestCosh(OperatorTest):
    @dtype_test
    def test_cosh_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = CoshModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_cosh_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(CoshModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_cosh_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(CoshModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(CoshModel(), (torch.randn(5, 5),), tester_factory)
        
        # 3D tensor
        self._test_op(CoshModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(CoshModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(CoshModel(), (torch.randn(2, 2, 2, 2, 2),), tester_factory)
        
    def test_cosh_specific_values(self, tester_factory: Callable) -> None:
        # Test with specific values
        self._test_op(
            CoshModel(), 
            (torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]),), 
            tester_factory
        )
        
    def test_cosh_symmetry(self, tester_factory: Callable) -> None:
        # Test symmetry property: cosh(-x) = cosh(x)
        self._test_op(
            CoshModel(), 
            (torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),), 
            tester_factory
        )
        
    def test_cosh_range(self, tester_factory: Callable) -> None:
        # Test with values in different ranges
        # Small values
        self._test_op(
            CoshModel(), 
            (torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),), 
            tester_factory
        )
        
        # Medium values
        self._test_op(
            CoshModel(), 
            (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),), 
            tester_factory
        )
        
        # Larger values (but not too large to avoid overflow)
        self._test_op(
            CoshModel(), 
            (torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0]),), 
            tester_factory
        )
        
    def test_cosh_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero values
        self._test_op(CoshModel(), (torch.zeros(5),), tester_factory)
        
        # Moderate values (cosh grows exponentially, so we avoid very large inputs)
        self._test_op(
            CoshModel(), 
            (torch.tensor([15.0, 20.0, -15.0, -20.0]),), 
            tester_factory
        )
        
        # Infinity and NaN
        self._test_op(
            CoshModel(), 
            (torch.tensor([float('inf'), float('-inf'), float('nan')]),), 
            tester_factory
        )
        
        # Single element tensor
        self._test_op(CoshModel(), (torch.tensor([1.5]),), tester_factory)
