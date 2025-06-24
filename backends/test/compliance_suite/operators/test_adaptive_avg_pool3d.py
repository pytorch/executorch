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
        output_size=(3, 3, 3),
    ):
        super().__init__()
        self.adaptive_avgpool = torch.nn.AdaptiveAvgPool3d(
            output_size=output_size,
        )
        
    def forward(self, x):
        return self.adaptive_avgpool(x)

@operator_test
class TestAdaptiveAvgPool3d(OperatorTest):
    @dtype_test
    def test_adaptive_avgpool3d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, depth, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 4, 4, 4) * 10).to(dtype),), tester_factory)
        
    def test_adaptive_avgpool3d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_adaptive_avgpool3d_output_size(self, tester_factory: Callable) -> None:
        # Test with different output sizes
        self._test_op(Model(output_size=1), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(output_size=2), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(output_size=(1, 2, 2)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        self._test_op(Model(output_size=(2, 1, 3)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_adaptive_avgpool3d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(output_size=(2, 2, 2)), (torch.randn(1, 1, 3, 3, 3),), tester_factory)
        self._test_op(Model(output_size=(3, 3, 3)), (torch.randn(2, 4, 5, 6, 7),), tester_factory)
        self._test_op(Model(output_size=(2, 3, 4)), (torch.randn(3, 2, 6, 8, 10),), tester_factory)
        
    def test_adaptive_avgpool3d_same_size(self, tester_factory: Callable) -> None:
        # Test when output size is the same as input size
        self._test_op(Model(output_size=(4, 4, 4)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
        
    def test_adaptive_avgpool3d_larger_size(self, tester_factory: Callable) -> None:
        # Test when output size is larger than input size (should be equivalent to identity)
        self._test_op(Model(output_size=(5, 5, 5)), (torch.randn(2, 3, 4, 4, 4),), tester_factory)
