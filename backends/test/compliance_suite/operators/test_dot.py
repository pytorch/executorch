# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class DotModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.dot(x, y)

@operator_test
class TestDot(OperatorTest):
    @dtype_test
    def test_dot_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype.is_complex:
            # Skip complex dtypes for now
            return
        
        model = DotModel().to(dtype)
        # Create two 1D vectors with the same size
        x = torch.rand(5).to(dtype)
        y = torch.rand(5).to(dtype)
        self._test_op(model, (x, y), tester_factory)
        
    def test_dot_basic(self, tester_factory: Callable) -> None:
        # Basic test with vectors of different sizes
        
        # Small vectors
        x = torch.randn(3)
        y = torch.randn(3)
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Medium vectors
        x = torch.randn(10)
        y = torch.randn(10)
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Large vectors
        x = torch.randn(100)
        y = torch.randn(100)
        self._test_op(DotModel(), (x, y), tester_factory)
        
    def test_dot_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Standard basis vectors (dot product should be 0)
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Parallel vectors (dot product should be |x|*|y|)
        x = torch.tensor([2.0, 4.0, 6.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Tensor with negative values
        x = torch.tensor([-1.0, -2.0, -3.0])
        y = torch.tensor([-4.0, -5.0, -6.0])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Tensor with mixed positive and negative values
        x = torch.tensor([-1.0, 2.0, -3.0])
        y = torch.tensor([4.0, -5.0, 6.0])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Tensor with fractional values
        x = torch.tensor([0.5, 1.5, 2.5])
        y = torch.tensor([3.5, 4.5, 5.5])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Integer tensor
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Vectors with known dot product
        # [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        self._test_op(DotModel(), (x, y), tester_factory)
        
    def test_dot_special_vectors(self, tester_factory: Callable) -> None:
        # Test with special vectors
        
        # Vectors with very large values
        x = torch.tensor([1e5, 2e5, 3e5])
        y = torch.tensor([4e5, 5e5, 6e5])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Vectors with very small values
        x = torch.tensor([1e-5, 2e-5, 3e-5])
        y = torch.tensor([4e-5, 5e-5, 6e-5])
        self._test_op(DotModel(), (x, y), tester_factory)
        
        # Vectors with mixed large and small values
        x = torch.tensor([1e5, 2e-5, 3e5])
        y = torch.tensor([4e-5, 5e5, 6e-5])
        self._test_op(DotModel(), (x, y), tester_factory)
