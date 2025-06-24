# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Union, Tuple

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
    def forward(self, x):
        return self.conv_transpose(x)

@operator_test
class TestConvTranspose1d(OperatorTest):
    @dtype_test
    def test_convtranspose1d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, in_channels, length)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 10) * 10).to(dtype),), tester_factory)
        
    def test_convtranspose1d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_kernel_size(self, tester_factory: Callable) -> None:
        # Test with different kernel sizes
        self._test_op(Model(kernel_size=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(kernel_size=5), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_stride(self, tester_factory: Callable) -> None:
        # Test with different stride values
        self._test_op(Model(stride=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 10),), tester_factory)
        self._test_op(Model(padding=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_output_padding(self, tester_factory: Callable) -> None:
        # Test with different output_padding values (requires stride > 1)
        self._test_op(Model(stride=2, output_padding=1), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_dilation(self, tester_factory: Callable) -> None:
        # Test with different dilation values
        self._test_op(Model(dilation=2), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_groups(self, tester_factory: Callable) -> None:
        # Test with groups=3 (in_channels and out_channels must be divisible by groups)
        self._test_op(Model(in_channels=6, out_channels=6, groups=3), (torch.randn(2, 6, 10),), tester_factory)
        
    def test_convtranspose1d_no_bias(self, tester_factory: Callable) -> None:
        # Test without bias
        self._test_op(Model(bias=False), (torch.randn(2, 3, 10),), tester_factory)
        
    def test_convtranspose1d_channels(self, tester_factory: Callable) -> None:
        # Test with different channel configurations
        self._test_op(Model(in_channels=1, out_channels=1), (torch.randn(2, 1, 10),), tester_factory)
        self._test_op(Model(in_channels=5, out_channels=10), (torch.randn(2, 5, 10),), tester_factory)
        