# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AtanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.atan(x)

@operator_test
class TestAtan(OperatorTest):
    @dtype_test
    def test_atan_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = AtanModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_atan_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(AtanModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_atan_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(AtanModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(AtanModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(AtanModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(AtanModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(AtanModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_atan_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values
        self._test_op(AtanModel(), (torch.randn(10, 10) * 0.1,), tester_factory)
        
        # Medium values
        self._test_op(AtanModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Large values
        self._test_op(AtanModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Very large values
        self._test_op(AtanModel(), (torch.randn(10, 10) * 100000,), tester_factory)
        
        # Specific values
        self._test_op(AtanModel(), (torch.tensor([-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]).view(7, 1),), tester_factory)
        
    def test_atan_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero
        self._test_op(AtanModel(), (torch.zeros(5, 5),), tester_factory)
        
        # Single-element tensor
        self._test_op(AtanModel(), (torch.tensor([-1.0]).view(1, 1),), tester_factory)
        self._test_op(AtanModel(), (torch.tensor([0.0]).view(1, 1),), tester_factory)
        self._test_op(AtanModel(), (torch.tensor([1.0]).view(1, 1),), tester_factory)
        
        # Infinity
        self._test_op(AtanModel(), (torch.tensor([float('inf'), -float('inf')]).view(2, 1),), tester_factory)
        
        # NaN
        self._test_op(AtanModel(), (torch.tensor([float('nan')]).view(1, 1),), tester_factory)
