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
        num_groups=2,
        num_channels=10,
        eps=1e-5,
        affine=True,
    ):
        super().__init__()
        self.gn = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )
        
    def forward(self, x):
        return self.gn(x)

@operator_test
class TestGroupNorm(OperatorTest):
    @dtype_test
    def test_groupnorm_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_channels, *) - can be any shape with num_channels as second dimension
        self._test_op(Model().to(dtype), ((torch.rand(3, 10, 5) * 10).to(dtype),), tester_factory)
        
    def test_groupnorm_2d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_channels)
        self._test_op(Model(), (torch.randn(5, 10),), tester_factory)
        
    def test_groupnorm_3d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_channels, length)
        self._test_op(Model(), (torch.randn(5, 10, 15),), tester_factory)
        
    def test_groupnorm_4d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_channels, height, width)
        self._test_op(Model(), (torch.randn(5, 10, 8, 8),), tester_factory)
        
    def test_groupnorm_5d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_channels, depth, height, width)
        self._test_op(Model(), (torch.randn(5, 10, 4, 4, 4),), tester_factory)
        
    def test_groupnorm_single_group(self, tester_factory: Callable) -> None:
        self._test_op(Model(num_groups=1), (torch.randn(5, 10, 8),), tester_factory)
        
    def test_groupnorm_channel_per_group(self, tester_factory: Callable) -> None:
        self._test_op(Model(num_groups=10), (torch.randn(5, 10, 8),), tester_factory)
        
    def test_groupnorm_custom_eps(self, tester_factory: Callable) -> None:
        self._test_op(Model(eps=1e3), (torch.randn(5, 10, 8),), tester_factory)
        
    def test_groupnorm_no_affine(self, tester_factory: Callable) -> None:
        self._test_op(Model(affine=False), (torch.randn(5, 10, 8),), tester_factory)
