# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SqrtModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.sqrt(x)

@operator_test
class TestSqrt(OperatorTest):
    @dtype_test
    def test_sqrt_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SqrtModel().to(dtype)
        # Use non-negative values only for sqrt
        self._test_op(model, (torch.rand(10, 10).to(dtype),), tester_factory)
        
    def test_sqrt_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with non-negative values
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 10,), tester_factory)
        
    def test_sqrt_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(SqrtModel(), (torch.rand(20),), tester_factory)
        
        # 2D tensor
        self._test_op(SqrtModel(), (torch.rand(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(SqrtModel(), (torch.rand(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(SqrtModel(), (torch.rand(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(SqrtModel(), (torch.rand(2, 2, 3, 4, 5),), tester_factory)
        
    def test_sqrt_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values close to zero
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 0.01,), tester_factory)
        
        # Values around 1
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 0.2 + 0.9,), tester_factory)
        
        # Perfect squares
        x = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])
        self._test_op(SqrtModel(), (x,), tester_factory)
        
        # Medium values
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 10,), tester_factory)
        
        # Large values
        self._test_op(SqrtModel(), (torch.rand(10, 10) * 1000,), tester_factory)
        
        # Very large values
        self._test_op(SqrtModel(), (torch.rand(5, 5) * 1e10,), tester_factory)
        
    def test_sqrt_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor
        self._test_op(SqrtModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.25, 0.5, 0.01])
        self._test_op(SqrtModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), 1.0, 4.0])
        self._test_op(SqrtModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, 4.0])
        self._test_op(SqrtModel(), (x,), tester_factory)
        
        # Values very close to zero
        x = torch.tensor([1e-10, 1e-20, 1e-30])
        self._test_op(SqrtModel(), (x,), tester_factory)
        
    def test_sqrt_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(SqrtModel(), (torch.tensor([0.0]),), tester_factory)
        self._test_op(SqrtModel(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(SqrtModel(), (torch.tensor([4.0]),), tester_factory)
        self._test_op(SqrtModel(), (torch.tensor([0.25]),), tester_factory)
