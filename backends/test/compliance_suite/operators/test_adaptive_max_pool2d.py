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
        output_size=(5, 5),
        return_indices=False,
    ):
        super().__init__()
        self.adaptive_maxpool = torch.nn.AdaptiveMaxPool2d(
            output_size=output_size,
            return_indices=return_indices,
        )
        
    def forward(self, x):
        return self.adaptive_maxpool(x)

@operator_test
class TestAdaptiveMaxPool2d(OperatorTest):
    @dtype_test
    def test_adaptive_maxpool2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(2, 3, 8, 8) * 10).to(dtype),), tester_factory)
        
    def test_adaptive_maxpool2d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_adaptive_maxpool2d_output_size(self, tester_factory: Callable) -> None:
        # Test with different output sizes
        self._test_op(Model(output_size=1), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(output_size=3), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(output_size=(1, 3)), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(Model(output_size=(3, 2)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_adaptive_maxpool2d_return_indices(self, tester_factory: Callable) -> None:
        # Test with return_indices=True
        self._test_op(Model(return_indices=True), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_adaptive_maxpool2d_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        self._test_op(Model(output_size=(3, 3)), (torch.randn(1, 1, 5, 5),), tester_factory)
        self._test_op(Model(output_size=(5, 5)), (torch.randn(2, 4, 10, 12),), tester_factory)
        self._test_op(Model(output_size=(4, 6)), (torch.randn(3, 2, 15, 20),), tester_factory)
        
    def test_adaptive_maxpool2d_same_size(self, tester_factory: Callable) -> None:
        # Test when output size is the same as input size
        self._test_op(Model(output_size=(8, 8)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_adaptive_maxpool2d_larger_size(self, tester_factory: Callable) -> None:
        # Test when output size is larger than input size (should be equivalent to identity)
        self._test_op(Model(output_size=(10, 10)), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_adaptive_maxpool2d_combinations(self, tester_factory: Callable) -> None:
        # Test with combinations of parameters
        self._test_op(Model(output_size=(3, 3), return_indices=True), (torch.randn(2, 3, 8, 8),), tester_factory)
