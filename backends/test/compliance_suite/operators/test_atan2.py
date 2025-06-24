# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Atan2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, x):
        return torch.atan2(y, x)

@operator_test
class TestAtan2(OperatorTest):
    @dtype_test
    def test_atan2_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = Atan2Model().to(dtype)
        self._test_op(model, (torch.randn(10, 10).to(dtype), torch.randn(10, 10).to(dtype)), tester_factory)
        
    def test_atan2_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Atan2Model(), (torch.randn(10, 10), torch.randn(10, 10)), tester_factory)
        
    def test_atan2_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(Atan2Model(), (torch.randn(20), torch.randn(20)), tester_factory)
        
        # 2D tensor
        self._test_op(Atan2Model(), (torch.randn(5, 10), torch.randn(5, 10)), tester_factory)
        
        # 3D tensor
        self._test_op(Atan2Model(), (torch.randn(3, 4, 5), torch.randn(3, 4, 5)), tester_factory)
        
        # 4D tensor
        self._test_op(Atan2Model(), (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)), tester_factory)
        
        # 5D tensor
        self._test_op(Atan2Model(), (torch.randn(2, 2, 3, 4, 5), torch.randn(2, 2, 3, 4, 5)), tester_factory)
        
    def test_atan2_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 0.1, torch.randn(10, 10) * 0.1), tester_factory)
        
        # Medium values
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 10, torch.randn(10, 10) * 10), tester_factory)
        
        # Large values
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 1000, torch.randn(10, 10) * 1000), tester_factory)
        
        # Very large values
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 100000, torch.randn(10, 10) * 100000), tester_factory)
        
        # Mixed values
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 0.1, torch.randn(10, 10) * 10), tester_factory)
        self._test_op(Atan2Model(), (torch.randn(10, 10) * 10, torch.randn(10, 10) * 0.1), tester_factory)
        
        # Specific values for quadrant testing
        y = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0]).view(8, 1)
        x = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0]).view(8, 1)
        self._test_op(Atan2Model(), (y, x), tester_factory)
        
    def test_atan2_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero denominator (x=0)
        self._test_op(Atan2Model(), (torch.tensor([1.0, -1.0, 0.0]).view(3, 1), torch.zeros(3, 1)), tester_factory)
        
        # Zero numerator (y=0)
        self._test_op(Atan2Model(), (torch.zeros(3, 1), torch.tensor([1.0, -1.0, 0.0]).view(3, 1)), tester_factory)
        
        # Single-element tensor
        self._test_op(Atan2Model(), (torch.tensor([1.0]).view(1, 1), torch.tensor([1.0]).view(1, 1)), tester_factory)
        self._test_op(Atan2Model(), (torch.tensor([0.0]).view(1, 1), torch.tensor([1.0]).view(1, 1)), tester_factory)
        self._test_op(Atan2Model(), (torch.tensor([1.0]).view(1, 1), torch.tensor([0.0]).view(1, 1)), tester_factory)
        
        # Infinity
        self._test_op(
            Atan2Model(), 
            (
                torch.tensor([float('inf'), -float('inf'), 1.0, -1.0]).view(4, 1),
                torch.tensor([1.0, 1.0, float('inf'), float('inf')]).view(4, 1)
            ), 
            tester_factory
        )
        
        # NaN
        self._test_op(
            Atan2Model(), 
            (
                torch.tensor([float('nan'), 1.0]).view(2, 1),
                torch.tensor([1.0, float('nan')]).view(2, 1)
            ), 
            tester_factory
        )
        
    def test_atan2_broadcasting(self, tester_factory: Callable) -> None:
        # Test broadcasting
        
        # Broadcasting y
        self._test_op(Atan2Model(), (torch.randn(1, 10), torch.randn(5, 10)), tester_factory)
        
        # Broadcasting x
        self._test_op(Atan2Model(), (torch.randn(5, 10), torch.randn(1, 10)), tester_factory)
        
        # Broadcasting both
        self._test_op(Atan2Model(), (torch.randn(5, 1), torch.randn(1, 10)), tester_factory)
