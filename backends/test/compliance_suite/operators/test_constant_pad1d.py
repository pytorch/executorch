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
    def __init__(
        self,
        padding=2,
        value=0.0,
    ):
        super().__init__()
        self.constant_pad = torch.nn.ConstantPad1d(
            padding=padding,
            value=value,
        )
        
    def forward(self, x):
        return self.constant_pad(x)

@operator_test
class TestConstantPad1d(OperatorTest):
    @dtype_test
    def test_constant_pad1d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, length)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 10) * 10).to(dtype),), tester_factory)
        
    def test_constant_pad1d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_constant_pad1d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=3), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=(1, 2)), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=(3, 1)), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_constant_pad1d_values(self, tester_factory: Callable) -> None:
        # Test with different constant values
        self._test_op(Model(value=1.0), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(value=-1.0), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(value=5.0), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_constant_pad1d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(), (torch.randn(1, 1, 5),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 4, 15),), tester_factory)
        
    def test_constant_pad1d_zero_padding(self, tester_factory: Callable) -> None:
        # Test with zero padding (should be equivalent to identity)
        self._test_op(Model(padding=0), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=(0, 0)), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_constant_pad1d_asymmetric_padding(self, tester_factory: Callable) -> None:
        # Test with asymmetric padding
        self._test_op(Model(padding=(0, 2)), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=(2, 0)), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_constant_pad1d_combinations(self, tester_factory: Callable) -> None:
        # Test with combinations of padding and values
        self._test_op(Model(padding=(1, 2), value=1.5), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=(3, 0), value=-0.5), (torch.randn(2, 3, 10),), tester_factory)
