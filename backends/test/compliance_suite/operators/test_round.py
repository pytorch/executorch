# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class RoundModel(torch.nn.Module):
    def __init__(self, decimals=None):
        super().__init__()
        self.decimals = decimals
        
    def forward(self, x):
        if self.decimals is not None:
            return torch.round(x, decimals=self.decimals)
        return torch.round(x)

@operator_test
class TestRound(OperatorTest):
    @dtype_test
    def test_round_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = RoundModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 10 - 5,), tester_factory)
        
    def test_round_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with fractional values
        self._test_op(RoundModel(), (torch.randn(10, 10) * 5,), tester_factory)
        
    def test_round_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(RoundModel(), (torch.randn(20) * 5,), tester_factory)
        
        # 2D tensor
        self._test_op(RoundModel(), (torch.randn(5, 10) * 5,), tester_factory)
        
        # 3D tensor
        self._test_op(RoundModel(), (torch.randn(3, 4, 5) * 5,), tester_factory)
        
        # 4D tensor
        self._test_op(RoundModel(), (torch.randn(2, 3, 4, 5) * 5,), tester_factory)
        
        # 5D tensor
        self._test_op(RoundModel(), (torch.randn(2, 2, 3, 4, 5) * 5,), tester_factory)
        
    def test_round_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small fractional values
        self._test_op(RoundModel(), (torch.randn(10, 10) * 0.1,), tester_factory)
        
        # Medium fractional values
        self._test_op(RoundModel(), (torch.randn(10, 10) * 5,), tester_factory)
        
        # Large fractional values
        self._test_op(RoundModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(RoundModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Values with specific fractional parts
        x = torch.arange(-5, 5, 0.5)  # [-5.0, -4.5, -4.0, ..., 4.0, 4.5]
        self._test_op(RoundModel(), (x,), tester_factory)
        
    def test_round_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Integer values (should remain unchanged)
        self._test_op(RoundModel(), (torch.arange(-5, 6).float(),), tester_factory)
        
        # Values exactly halfway between integers (should round to even)
        x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        self._test_op(RoundModel(), (x,), tester_factory)
        
        # Values slightly above and below halfway
        x = torch.tensor([-2.51, -2.49, -1.51, -1.49, -0.51, -0.49, 0.49, 0.51, 1.49, 1.51, 2.49, 2.51])
        self._test_op(RoundModel(), (x,), tester_factory)
        
        # Zero tensor
        self._test_op(RoundModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.4, -1.4])
        self._test_op(RoundModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.4, -1.4])
        self._test_op(RoundModel(), (x,), tester_factory)
        
        # Very large values (where fractional part becomes insignificant)
        x = torch.tensor([1e10, 1e10 + 0.4, 1e10 + 0.6])
        self._test_op(RoundModel(), (x,), tester_factory)
        
        # Very small values close to zero
        x = torch.tensor([-0.1, -0.01, -0.001, 0.001, 0.01, 0.1])
        self._test_op(RoundModel(), (x,), tester_factory)
        
    def test_round_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(RoundModel(), (torch.tensor([1.4]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([1.5]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([1.6]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([-1.4]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([-1.5]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([-1.6]),), tester_factory)
        self._test_op(RoundModel(), (torch.tensor([0.0]),), tester_factory)
        
    def test_round_decimals(self, tester_factory: Callable) -> None:
        # Test with different decimal places
        
        # Round to 1 decimal place
        x = torch.tensor([1.44, 1.45, 1.46, -1.44, -1.45, -1.46])
        self._test_op(RoundModel(decimals=1), (x,), tester_factory)
        
        # Round to 2 decimal places
        x = torch.tensor([1.444, 1.445, 1.446, -1.444, -1.445, -1.446])
        self._test_op(RoundModel(decimals=2), (x,), tester_factory)
        
        # Round to negative decimal places (tens)
        x = torch.tensor([14.4, 15.5, 16.6, -14.4, -15.5, -16.6])
        self._test_op(RoundModel(decimals=-1), (x,), tester_factory)
        
        # Round to negative decimal places (hundreds)
        x = torch.tensor([144.4, 155.5, 166.6, -144.4, -155.5, -166.6])
        self._test_op(RoundModel(decimals=-2), (x,), tester_factory)
        
    def test_round_decimals_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases with decimal places
        
        # Very small values with positive decimals
        x = torch.tensor([0.0001, 0.00015, 0.0002, -0.0001, -0.00015, -0.0002])
        self._test_op(RoundModel(decimals=4), (x,), tester_factory)
        
        # Very large values with negative decimals
        x = torch.tensor([12345.6, 12350.0, 12354.9, -12345.6, -12350.0, -12354.9])
        self._test_op(RoundModel(decimals=-2), (x,), tester_factory)
        
        # Zero with various decimal places
        x = torch.zeros(5)
        self._test_op(RoundModel(decimals=2), (x,), tester_factory)
        self._test_op(RoundModel(decimals=-2), (x,), tester_factory)
        
        # Infinity and NaN with various decimal places
        x = torch.tensor([float('inf'), float('-inf'), float('nan')])
        self._test_op(RoundModel(decimals=2), (x,), tester_factory)
        self._test_op(RoundModel(decimals=-2), (x,), tester_factory)
        
        # Values exactly at the rounding threshold for different decimal places
        x = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        self._test_op(RoundModel(decimals=1), (x,), tester_factory)
        
        # Negative values exactly at the rounding threshold
        x = torch.tensor([-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95])
        self._test_op(RoundModel(decimals=1), (x,), tester_factory)
