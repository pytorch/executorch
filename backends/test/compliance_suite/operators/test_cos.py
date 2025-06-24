# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class CosModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.cos(x)

@operator_test
class TestCos(OperatorTest):
    @dtype_test
    def test_cos_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = CosModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_cos_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(CosModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_cos_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(CosModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(CosModel(), (torch.randn(5, 5),), tester_factory)
        
        # 3D tensor
        self._test_op(CosModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(CosModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(CosModel(), (torch.randn(2, 2, 2, 2, 2),), tester_factory)
        
    def test_cos_specific_values(self, tester_factory: Callable) -> None:
        # Test with specific values
        self._test_op(
            CosModel(), 
            (torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]),), 
            tester_factory
        )
        
    def test_cos_full_period(self, tester_factory: Callable) -> None:
        # Test with values that cover the full period of cosine
        self._test_op(
            CosModel(), 
            (torch.tensor([0.0, 3.14159/4, 3.14159/2, 3.14159, 3.14159*1.5, 3.14159*2]),), 
            tester_factory
        )
        
        # Test with key values where cos has specific outputs
        self._test_op(
            CosModel(), 
            (torch.tensor([0.0, 3.14159/2, 3.14159, 3.14159*3/2, 3.14159*2]),), 
            tester_factory
        )
        
    def test_cos_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero values
        self._test_op(CosModel(), (torch.zeros(5),), tester_factory)
        
        # Large values
        self._test_op(
            CosModel(), 
            (torch.tensor([1e3, 1e4, 1e5, -1e3, -1e4, -1e5]),), 
            tester_factory
        )
        
        # Infinity and NaN
        self._test_op(
            CosModel(), 
            (torch.tensor([float('inf'), float('-inf'), float('nan')]),), 
            tester_factory
        )
        
        # Single element tensor
        self._test_op(CosModel(), (torch.tensor([3.14159/4]),), tester_factory)
        
