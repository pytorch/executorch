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
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        
    def forward(self, x):
        return self.conv(x)

@operator_test
class TestConv3d(OperatorTest):
    @dtype_test
    def test_conv3d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, in_channels, depth, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 4, 4, 4) * 10).to(dtype),), tester_factory)
        
    def test_conv3d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_conv3d_kernel_size(self, tester_factory: Callable) -> None:
        # Test with different kernel sizes
        self._test_op(Model(kernel_size=1), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(kernel_size=(1, 3, 3)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_conv3d_stride(self, tester_factory: Callable) -> None:
        # Test with different stride values
        self._test_op(Model(stride=2), (torch.randn(2, 3, 6, 6, 6),), tester_factory)
        self._test_op(Model(stride=(1, 2, 2)), (torch.randn(2, 3, 4, 6, 6),), tester_factory)
        
    def test_conv3d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(padding=(0, 1, 1)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_conv3d_dilation(self, tester_factory: Callable) -> None:
        # Test with different dilation values
        self._test_op(Model(dilation=2), (torch.randn(2, 3, 6, 6, 6),), tester_factory)
        self._test_op(Model(dilation=(1, 2, 2)), (torch.randn(2, 3, 4, 6, 6),), tester_factory)
        
    def test_conv3d_groups(self, tester_factory: Callable) -> None:
        # Test with groups=3 (in_channels must be divisible by groups)
        self._test_op(Model(in_channels=6, out_channels=6, groups=3), (torch.randn(2, 6, 4, 4, 4),), tester_factory)
        
    def test_conv3d_no_bias(self, tester_factory: Callable) -> None:
        # Test without bias
        self._test_op(Model(bias=False), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_conv3d_padding_modes(self, tester_factory: Callable) -> None:
        # Test different padding modes
        for mode in ["zeros", "reflect", "replicate", "circular"]:
            self._test_op(Model(padding=1, padding_mode=mode), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
            
    def test_conv3d_channels(self, tester_factory: Callable) -> None:
        # Test with different channel configurations
        self._test_op(Model(in_channels=1, out_channels=1), (torch.randn(2, 1, 4, 4, 4),), tester_factory)
        self._test_op(Model(in_channels=5, out_channels=10), (torch.randn(2, 5, 4, 4, 4),), tester_factory)
        
    def test_conv3d_different_spatial_dims(self, tester_factory: Callable) -> None:
        # Test with different depth, height, and width
        self._test_op(Model(), (torch.randn(2, 3, 3, 4, 5),), tester_factory)
