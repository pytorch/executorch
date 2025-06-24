# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ClampModel(torch.nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)

@operator_test
class TestClamp(OperatorTest):
    @dtype_test
    def test_clamp_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = ClampModel(min_val=-0.5, max_val=0.5).to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), tester_factory)
        
    def test_clamp_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with values outside the clamp range
        self._test_op(ClampModel(min_val=-0.5, max_val=0.5), (torch.randn(10, 10),), tester_factory)
        
    def test_clamp_min_only(self, tester_factory: Callable) -> None:
        # Test with only min value specified
        self._test_op(ClampModel(min_val=0.0), (torch.randn(10, 10),), tester_factory)
        
    def test_clamp_max_only(self, tester_factory: Callable) -> None:
        # Test with only max value specified
        self._test_op(ClampModel(max_val=0.0), (torch.randn(10, 10),), tester_factory)
        
    def test_clamp_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        model = ClampModel(min_val=-1.0, max_val=1.0)
        
        # 1D tensor
        self._test_op(model, (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(model, (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(model, (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(model, (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(model, (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_clamp_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values with narrow clamp range
        self._test_op(ClampModel(min_val=-0.01, max_val=0.01), (torch.randn(10, 10) * 0.1,), tester_factory)
        
        # Large values with wide clamp range
        self._test_op(ClampModel(min_val=-100, max_val=100), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(ClampModel(min_val=-5, max_val=5), (torch.randn(10, 10) * 10,), tester_factory)
        
        # All values within clamp range
        self._test_op(ClampModel(min_val=-10, max_val=10), (torch.randn(10, 10),), tester_factory)
        
        # All values outside clamp range (below min)
        self._test_op(ClampModel(min_val=1.0, max_val=2.0), (torch.randn(10, 10) - 10,), tester_factory)
        
        # All values outside clamp range (above max)
        self._test_op(ClampModel(min_val=-2.0, max_val=-1.0), (torch.randn(10, 10) + 10,), tester_factory)
        
    def test_clamp_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor
        self._test_op(ClampModel(min_val=-1.0, max_val=1.0), (torch.zeros(10, 10),), tester_factory)
        
        # Min equals max
        self._test_op(ClampModel(min_val=0.0, max_val=0.0), (torch.randn(10, 10),), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0])
        self._test_op(ClampModel(min_val=-2.0, max_val=2.0), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, -1.0])
        self._test_op(ClampModel(min_val=-2.0, max_val=2.0), (x,), tester_factory)
        
        # Values at exactly min/max bounds
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        self._test_op(ClampModel(min_val=-0.5, max_val=0.5), (x,), tester_factory)
        
    def test_clamp_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        model = ClampModel(min_val=-1.0, max_val=1.0)
        self._test_op(model, (torch.tensor([-5.0]),), tester_factory)
        self._test_op(model, (torch.tensor([5.0]),), tester_factory)
        self._test_op(model, (torch.tensor([0.0]),), tester_factory)
