# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class TruncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.trunc(x)

@operator_test
class TestTrunc(OperatorTest):
    @dtype_test
    def test_trunc_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = TruncModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 10 - 5,), tester_factory)
        
    def test_trunc_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 5,), tester_factory)
        
    def test_trunc_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(TruncModel(), (torch.randn(20) * 5,), tester_factory)
        
        # 2D tensor
        self._test_op(TruncModel(), (torch.randn(5, 10) * 5,), tester_factory)
        
        # 3D tensor
        self._test_op(TruncModel(), (torch.randn(3, 4, 5) * 5,), tester_factory)
        
        # 4D tensor
        self._test_op(TruncModel(), (torch.randn(2, 3, 4, 5) * 5,), tester_factory)
        
        # 5D tensor
        self._test_op(TruncModel(), (torch.randn(2, 2, 3, 4, 5) * 5,), tester_factory)
        
    def test_trunc_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 0.1,), tester_factory)
        
        # Medium fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 5,), tester_factory)
        
        # Large fractional values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(TruncModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Values with specific fractional parts
        x = torch.arange(-5, 5, 0.5)  # [-5.0, -4.5, -4.0, ..., 4.0, 4.5]
        self._test_op(TruncModel(), (x,), tester_factory)
        
    def test_trunc_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Integer values (should remain unchanged)
        self._test_op(TruncModel(), (torch.arange(-5, 6).float(),), tester_factory)
        
        # Values with different fractional parts
        x = torch.tensor([-2.9, -2.5, -2.1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 2.1, 2.5, 2.9])
        self._test_op(TruncModel(), (x,), tester_factory)
        
        # Zero tensor
        self._test_op(TruncModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.4, -1.4])
        self._test_op(TruncModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.4, -1.4])
        self._test_op(TruncModel(), (x,), tester_factory)
        
        # Very large values (where fractional part becomes insignificant)
        x = torch.tensor([1e10, 1e10 + 0.4, 1e10 + 0.6])
        self._test_op(TruncModel(), (x,), tester_factory)
        
        # Very small values close to zero
        x = torch.tensor([-0.1, -0.01, -0.001, 0.001, 0.01, 0.1])
        self._test_op(TruncModel(), (x,), tester_factory)
        
    def test_trunc_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(TruncModel(), (torch.tensor([1.4]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([1.5]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([1.6]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([-1.4]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([-1.5]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([-1.6]),), tester_factory)
        self._test_op(TruncModel(), (torch.tensor([0.0]),), tester_factory)
