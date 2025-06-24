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
    ):
        super().__init__()
        self.replication_pad = torch.nn.ReplicationPad3d(
            padding=padding,
        )
        
    def forward(self, x):
        return self.replication_pad(x)

@operator_test
class TestReplicationPad3d(OperatorTest):
    @dtype_test
    def test_replication_pad3d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, depth, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 4, 4, 4) * 10).to(dtype),), tester_factory)
        
    def test_replication_pad3d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_replication_pad3d_padding(self, tester_factory: Callable) -> None:
        # Test with different padding values
        self._test_op(Model(padding=1), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(padding=(1, 1, 1, 1, 1, 1)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(padding=(1, 2, 1, 2, 1, 2)), (torch.randn(2, 3, 6, 6, 6),), tester_factory)
        
    def test_replication_pad3d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(padding=1), (torch.randn(1, 1, 3, 3, 3),), tester_factory)
        self._test_op(Model(padding=1), (torch.randn(2, 4, 5, 6, 7),), tester_factory)
        
    def test_replication_pad3d_zero_padding(self, tester_factory: Callable) -> None:
        # Test with zero padding (should be equivalent to identity)
        self._test_op(Model(padding=0), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(padding=(0, 0, 0, 0, 0, 0)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_replication_pad3d_asymmetric_padding(self, tester_factory: Callable) -> None:
        # Test with asymmetric padding
        self._test_op(Model(padding=(0, 2, 1, 0, 1, 0)), (torch.randn(2, 3, 5, 5, 5),), tester_factory)
        self._test_op(Model(padding=(1, 0, 0, 2, 2, 0)), (torch.randn(2, 3, 5, 5, 5),), tester_factory)
