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
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        
    def forward(self, x):
        return self.maxpool(x)

@operator_test
class TestMaxPool1d(OperatorTest):
    @dtype_test
    def test_maxpool1d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, length)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 10) * 10).to(dtype),), tester_factory)
        
    def test_maxpool1d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_kernel_size(self, tester_factory: Callable) -> None:
        # Test with different kernel sizes
        self._test_op(Model(kernel_size=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=5), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_stride(self, tester_factory: Callable) -> None:
        # Test with different stride values
        self._test_op(Model(stride=2), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(stride=3), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_dilation(self, tester_factory: Callable) -> None:
        # Test with different dilation values
        self._test_op(Model(dilation=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_ceil_mode(self, tester_factory: Callable) -> None:
        # Test with ceil_mode=True
        self._test_op(Model(ceil_mode=True), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_return_indices(self, tester_factory: Callable) -> None:
        # Test with return_indices=True
        # Note: This might need special handling as it returns a tuple
        self._test_op(Model(return_indices=True), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_maxpool1d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(), (torch.randn(1, 1, 5),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 4, 15),), tester_factory)
        
    def test_maxpool1d_combinations(self, tester_factory: Callable) -> None:
        # Test with combinations of parameters
        self._test_op(Model(kernel_size=2, stride=2, padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=3, stride=2, dilation=2), (torch.randn(2, 3, 15),), tester_factory)
        self._test_op(Model(kernel_size=2, stride=2, ceil_mode=True), (torch.randn(2, 3, 10),), tester_factory)
