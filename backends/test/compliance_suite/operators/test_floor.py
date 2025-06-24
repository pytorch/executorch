# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class FloorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.floor(x)

@operator_test
class TestFloor(OperatorTest):
    @dtype_test
    def test_floor_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = FloorModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), tester_factory)
        
    def test_floor_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with fractional values
        self._test_op(FloorModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_floor_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(FloorModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(FloorModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(FloorModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(FloorModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(FloorModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_floor_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small fractional values
        self._test_op(FloorModel(), (torch.rand(10, 10) * 0.01,), tester_factory)
        
        # Large fractional values
        self._test_op(FloorModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(FloorModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # Values with specific fractional parts
        self._test_op(FloorModel(), (torch.arange(0, 10, 0.5).reshape(4, 5),), tester_factory)
        
        # Values close to integers
        x = torch.randn(10, 10)
        x = x.round() + torch.rand(10, 10) * 0.01
        self._test_op(FloorModel(), (x,), tester_factory)
        
    def test_floor_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Integer values
        self._test_op(FloorModel(), (torch.arange(10).reshape(2, 5).float(),), tester_factory)
        
        # Zero tensor
        self._test_op(FloorModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, -1.0])
        self._test_op(FloorModel(), (x,), tester_factory)
        
        # Values just below integers
        x = torch.arange(10).float() - 0.01
        self._test_op(FloorModel(), (x,), tester_factory)
        
        # Values just above integers
        x = torch.arange(10).float() + 0.01
        self._test_op(FloorModel(), (x,), tester_factory)
        
    def test_floor_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(FloorModel(), (torch.tensor([1.5]),), tester_factory)
        self._test_op(FloorModel(), (torch.tensor([-1.5]),), tester_factory)
        self._test_op(FloorModel(), (torch.tensor([0.0]),), tester_factory)
