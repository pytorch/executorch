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
        num_features=10,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    ):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )
        
    def forward(self, x):
        return self.bn(x)

@operator_test
class TestBatchNorm2d(OperatorTest):
    @dtype_test
    def test_batchnorm2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_features, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(3, 10, 8, 8) * 10).to(dtype),), tester_factory)
        
    def test_batchnorm2d_4d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_features, height, width)
        self._test_op(Model(), (torch.randn(5, 10, 8, 8),), tester_factory)
        
    def test_batchnorm2d_small_spatial_dims(self, tester_factory: Callable) -> None:
        # Test with small spatial dimensions
        self._test_op(Model(), (torch.randn(5, 10, 2, 2),), tester_factory)
        
    def test_batchnorm2d_different_spatial_dims(self, tester_factory: Callable) -> None:
        # Test with different height and width
        self._test_op(Model(), (torch.randn(5, 10, 4, 8),), tester_factory)
        
    def test_batchnorm2d_custom_eps(self, tester_factory: Callable) -> None:
        self._test_op(Model(eps=1e3), (torch.randn(5, 10, 8, 8),), tester_factory)
        
    def test_batchnorm2d_no_affine(self, tester_factory: Callable) -> None:
        self._test_op(Model(affine=False), (torch.randn(5, 10, 8, 8),), tester_factory)
