# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Model(torch.nn.Module):
    def forward(self, x):
        # softmax2d is equivalent to softmax with dim=1 for 4D inputs
        return torch.nn.functional.softmax(x, dim=1)

@operator_test
class TestSoftmax2d(OperatorTest):
    @dtype_test
    def test_softmax2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input must be 4D (N, C, H, W)
        self._test_op(Model(), ((torch.rand(2, 3, 4, 5) * 100).to(dtype),), tester_factory)
        
    def test_softmax2d_f32_various_shapes(self, tester_factory: Callable) -> None:
        # Test with different shapes
        self._test_op(Model(), (torch.randn(1, 3, 8, 8),), tester_factory)
        
    def test_softmax2d_f32_single_channel(self, tester_factory: Callable) -> None:
        # Test with single channel (C=1)
        self._test_op(Model(), (torch.randn(2, 1, 4, 4),), tester_factory)
        
    def test_softmax2d_f32_many_channels(self, tester_factory: Callable) -> None:
        # Test with many channels
        self._test_op(Model(), (torch.randn(2, 16, 4, 4),), tester_factory)
    
    def test_softmax2d_f32_single_batch(self, tester_factory: Callable) -> None:
        # Test with single batch (N=1)
        self._test_op(Model(), (torch.randn(1, 3, 4, 4),), tester_factory)
        
    def test_softmax2d_f32_large_values(self, tester_factory: Callable) -> None:
        # Test with large values to check numerical stability
        x = torch.zeros(2, 3, 2, 2)
        x[:, 0] = 1000.0  # First channel has large positive values
        x[:, 1] = 0.0     # Second channel has zeros
        x[:, 2] = -1000.0 # Third channel has large negative values
        self._test_op(Model(), (x,), tester_factory)
