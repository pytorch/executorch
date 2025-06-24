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
        kernel_size=3,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        
    def forward(self, x):
        return self.avgpool(x)

@operator_test
class TestAvgPool1d(OperatorTest):
    @dtype_test
    def test_avgpool1d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, length)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 10) * 10).to(dtype),), tester_factory)
        
    def test_avgpool1d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_kernel_size(self, tester_factory: Callable) -> None:
        # Test with different kernel sizes
        self._test_op(Model(kernel_size=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=5), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_stride(self, tester_factory: Callable) -> None:
        # Test with different stride values
        self._test_op(Model(stride=2), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(stride=3), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_ceil_mode(self, tester_factory: Callable) -> None:
        # Test with ceil_mode=True
        self._test_op(Model(ceil_mode=True), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_count_include_pad(self, tester_factory: Callable) -> None:
        # Test with count_include_pad=False
        self._test_op(Model(padding=1, count_include_pad=False), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_avgpool1d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(), (torch.randn(1, 1, 5),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 4, 15),), tester_factory)
        
    def test_avgpool1d_combinations(self, tester_factory: Callable) -> None:
        # Test with combinations of parameters
        self._test_op(Model(kernel_size=2, stride=2, padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=3, stride=2, padding=1, ceil_mode=True), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=2, stride=2, padding=1, count_include_pad=False), (torch.randn(2, 3, 10),), tester_factory)
