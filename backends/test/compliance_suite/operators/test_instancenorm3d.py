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
        affine=False,
    ):
        super().__init__()
        self.in3d = torch.nn.InstanceNorm3d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )
        
    def forward(self, x):
        return self.in3d(x)

@operator_test
class TestInstanceNorm3d(OperatorTest):
    @dtype_test
    def test_instancenorm3d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_features, depth, height, width)
        self._test_op(Model().to(dtype), ((torch.rand(3, 10, 4, 4, 4) * 10).to(dtype),), tester_factory)
        
    def test_instancenorm3d_5d_input(self, tester_factory: Callable) -> None:
        # Input shape: (batch_size, num_features, depth, height, width)
        self._test_op(Model(), (torch.randn(5, 10, 4, 4, 4),), tester_factory)
        
    def test_instancenorm3d_different_features(self, tester_factory: Callable) -> None:
        # Test with different number of features
        self._test_op(Model(num_features=5), (torch.randn(5, 5, 4, 4, 4),), tester_factory)
        
    def test_instancenorm3d_different_spatial_dims(self, tester_factory: Callable) -> None:
        # Test with different depth, height, and width
        self._test_op(Model(), (torch.randn(5, 10, 3, 6, 8),), tester_factory)
        
    def test_instancenorm3d_custom_eps(self, tester_factory: Callable) -> None:
        self._test_op(Model(eps=1e3), (torch.randn(5, 10, 4, 4, 4),), tester_factory)
        
    def test_instancenorm3d_with_affine(self, tester_factory: Callable) -> None:
        # InstanceNorm has affine=False by default, test with affine=True
        self._test_op(Model(affine=True), (torch.randn(5, 10, 4, 4, 4),), tester_factory)
