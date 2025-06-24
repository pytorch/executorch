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
        in_features=10,
        out_features=5,
        bias=True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        
    def forward(self, x):
        return self.linear(x)

@operator_test
class TestLinear(OperatorTest):
    @dtype_test
    def test_linear_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, in_features)
        model = Model().to(dtype)
        self._test_op(model, ((torch.rand(2, 10) * 10).to(dtype),), tester_factory)

    @dtype_test
    def test_linear_no_bias_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, in_features)
        model = Model(bias=False).to(dtype)
        self._test_op(model, ((torch.rand(2, 10) * 10).to(dtype),), tester_factory)
        
    def test_linear_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(Model(), (torch.randn(2, 10),), tester_factory)
        
    def test_linear_feature_sizes(self, tester_factory: Callable) -> None:
        # Test with different input and output feature sizes
        self._test_op(Model(in_features=5, out_features=3), (torch.randn(2, 5),), tester_factory)
        self._test_op(Model(in_features=20, out_features=10), (torch.randn(2, 20),), tester_factory)
        self._test_op(Model(in_features=100, out_features=1), (torch.randn(2, 100),), tester_factory)
        self._test_op(Model(in_features=1, out_features=100), (torch.randn(2, 1),), tester_factory)
        
    def test_linear_no_bias(self, tester_factory: Callable) -> None:
        # Test without bias
        self._test_op(Model(bias=False), (torch.randn(2, 10),), tester_factory)
        self._test_op(Model(in_features=20, out_features=15, bias=False), (torch.randn(2, 20),), tester_factory)
        
    def test_linear_batch_sizes(self, tester_factory: Callable) -> None:
        # Test with different batch sizes
        self._test_op(Model(), (torch.randn(1, 10),), tester_factory)
        self._test_op(Model(), (torch.randn(5, 10),), tester_factory)
        self._test_op(Model(), (torch.randn(100, 10),), tester_factory)
        
    def test_linear_unbatched(self, tester_factory: Callable) -> None:
        # Test with unbatched input (just features)
        self._test_op(Model(), (torch.randn(10),), tester_factory)
        
    def test_linear_multi_dim_input(self, tester_factory: Callable) -> None:
        # Test with multi-dimensional input
        # For multi-dimensional inputs, the linear transformation is applied to the last dimension
        self._test_op(Model(), (torch.randn(3, 4, 10),), tester_factory)
        self._test_op(Model(), (torch.randn(2, 3, 4, 10),), tester_factory)
        