# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AsinhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.asinh(x)

@operator_test
class TestAsinh(OperatorTest):
    @dtype_test
    def test_asinh_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        # Input can be any real number for asinh
        model = AsinhModel().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype),), tester_factory)
        
    def test_asinh_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(AsinhModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_asinh_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(AsinhModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(AsinhModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(AsinhModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(AsinhModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(AsinhModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_asinh_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small negative values
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 0.1 - 0.1,), tester_factory)
        
        # Small positive values
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 0.1 + 0.1,), tester_factory)
        
        # Values around zero
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 0.01,), tester_factory)
        
        # Medium values
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Large values
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 100,), tester_factory)
        
        # Very large values
        self._test_op(AsinhModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Specific values
        self._test_op(AsinhModel(), (torch.tensor([-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]).view(7, 1),), tester_factory)
        
    def test_asinh_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero
        self._test_op(AsinhModel(), (torch.zeros(5, 5),), tester_factory)
        
        # Empty tensor
        self._test_op(AsinhModel(), (torch.randn(0, 10),), tester_factory)
        
        # Single-element tensor
        self._test_op(AsinhModel(), (torch.tensor([-1.0]).view(1, 1),), tester_factory)
        self._test_op(AsinhModel(), (torch.tensor([0.0]).view(1, 1),), tester_factory)
        self._test_op(AsinhModel(), (torch.tensor([1.0]).view(1, 1),), tester_factory)
        
        # Infinity
        self._test_op(AsinhModel(), (torch.tensor([float('inf'), -float('inf')]).view(2, 1),), tester_factory)
        
        # NaN
        self._test_op(AsinhModel(), (torch.tensor([float('nan')]).view(1, 1),), tester_factory)
