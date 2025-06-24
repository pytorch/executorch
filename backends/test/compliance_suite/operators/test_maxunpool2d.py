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
    ):
        super().__init__()
        # First, we need a MaxPool2d to get the indices
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True,
        )
        # Then, we use MaxUnpool2d to invert the operation
        self.maxunpool = torch.nn.MaxUnpool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        
    def forward(self, x):
        # Get the pooled output and indices from MaxPool2d
        pooled_output, indices = self.maxpool(x)
        # Use MaxUnpool2d to reconstruct the input
        return self.maxunpool(pooled_output, indices, output_size=x.size())

@operator_test
class TestMaxUnpool2d(OperatorTest):
    @dtype_test
    def test_maxunpool2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 8, 8) * 10).to(dtype),), tester_factory)
        
    def test_maxunpool2d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_maxunpool2d_kernel_size(self, tester_factory: Callable) -> None:
        # Test with different kernel sizes
        self._test_op(Model(kernel_size=1), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(kernel_size=2), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(kernel_size=(2, 3)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_maxunpool2d_stride(self, tester_factory: Callable) -> None:
        # Test with different stride values
        self._test_op(Model(stride=1), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(stride=2), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(stride=(2, 1)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_maxunpool2d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=0), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=1), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(padding=(1, 0)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_maxunpool2d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(), (torch.randn(1, 1, 5, 5),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 4, 10, 12),), tester_factory)
        
    def test_maxunpool2d_combinations(self, tester_factory: Callable) -> None:
        # Test with combinations of parameters
        self._test_op(Model(kernel_size=2, stride=2, padding=0), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(kernel_size=3, stride=2, padding=1), (torch.randn(2, 3, 9, 9),), tester_factory)
        self._test_op(Model(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)), (torch.randn(2, 3, 8, 8),), tester_factory)
