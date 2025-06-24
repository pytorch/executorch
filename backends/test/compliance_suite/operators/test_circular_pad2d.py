# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class CircularPad2d(torch.nn.Module):
    def __init__(
        self,
        padding: Union[int, Tuple[int, int, int, int]],
    ):
        super().__init__()
        if isinstance(padding, int):
            # Convert to tuple (left, right, top, bottom)
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding
        
    def forward(self, x):
        # F.pad expects (left, right, top, bottom) for 2D padding
        return F.pad(x, self.padding, mode='circular')

class Model(torch.nn.Module):
    def __init__(
        self,
        padding=2,
    ):
        super().__init__()
        self.circular_pad = CircularPad2d(padding=padding)
        
    def forward(self, x):
        return self.circular_pad(x)

@operator_test
class TestCircularPad2d(OperatorTest):
    @dtype_test
    def test_circular_pad2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 8, 8) * 10).to(dtype),), tester_factory)
        
    def test_circular_pad2d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_circular_pad2d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=3), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=(1, 2, 1, 2)), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=(3, 1, 2, 0)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_circular_pad2d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(), (torch.randn(1, 1, 5, 5),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 4, 10, 12),), tester_factory)
        
    def test_circular_pad2d_zero_padding(self, tester_factory: Callable) -> None:
        # Test with zero padding (should be equivalent to identity)
        self._test_op(Model(padding=0), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=(0, 0, 0, 0)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_circular_pad2d_asymmetric_padding(self, tester_factory: Callable) -> None:
        # Test with asymmetric padding
        self._test_op(Model(padding=(0, 2, 1, 0)), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=(2, 0, 0, 3)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_circular_pad2d_large_padding(self, tester_factory: Callable) -> None:
        # Test with padding larger than input size (should wrap around multiple times)
        self._test_op(Model(padding=(10, 10, 10, 10)), (torch.randn(2, 3, 8, 8),), tester_factory)
