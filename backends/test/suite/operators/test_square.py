# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SquareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.square(x)

@operator_test
class TestSquare(OperatorTest):
    @dtype_test
    def test_square_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SquareModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), tester_factory)
        
    def test_square_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with positive and negative values
        self._test_op(SquareModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_square_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(SquareModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(SquareModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(SquareModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(SquareModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(SquareModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_square_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values
        self._test_op(SquareModel(), (torch.randn(10, 10) * 0.01,), tester_factory)
        
        # Values around 1
        self._test_op(SquareModel(), (torch.randn(10, 10) * 0.2 + 0.9,), tester_factory)
        
        # Medium values
        self._test_op(SquareModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Large values (be careful with overflow)
        self._test_op(SquareModel(), (torch.randn(10, 10) * 100,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(SquareModel(), (torch.randn(10, 10) * 5,), tester_factory)
        
        # All positive values
        self._test_op(SquareModel(), (torch.rand(10, 10) * 5,), tester_factory)
        
        # All negative values
        self._test_op(SquareModel(), (torch.rand(10, 10) * -5,), tester_factory)
        
        # Values close to zero
        self._test_op(SquareModel(), (torch.randn(10, 10) * 1e-5,), tester_factory)
        
        # Integer values
        x = torch.arange(-5, 6).float()  # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        self._test_op(SquareModel(), (x,), tester_factory)
        
    def test_square_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor
        self._test_op(SquareModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        self._test_op(SquareModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0])
        self._test_op(SquareModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, -1.0])
        self._test_op(SquareModel(), (x,), tester_factory)
        
        # Very large values (close to overflow for some dtypes)
        x = torch.tensor([1e10, -1e10])
        self._test_op(SquareModel(), (x,), tester_factory)
        
        # Very small values (close to underflow)
        x = torch.tensor([1e-10, -1e-10])
        self._test_op(SquareModel(), (x,), tester_factory)
        
    def test_square_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(SquareModel(), (torch.tensor([-5.0]),), tester_factory)
        self._test_op(SquareModel(), (torch.tensor([5.0]),), tester_factory)
        self._test_op(SquareModel(), (torch.tensor([0.0]),), tester_factory)
        self._test_op(SquareModel(), (torch.tensor([0.5]),), tester_factory)
        self._test_op(SquareModel(), (torch.tensor([-0.5]),), tester_factory)
