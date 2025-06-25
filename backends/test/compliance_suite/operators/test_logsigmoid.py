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
    def forward(self, x):
        return torch.nn.functional.logsigmoid(x)

@operator_test
class TestLogSigmoid(OperatorTest):
    @dtype_test
    def test_logsigmoid_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),), tester_factory)
        
    def test_logsigmoid_f32_single_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(20),), tester_factory)

    def test_logsigmoid_f32_multi_dim(self, tester_factory: Callable) -> None:
        self._test_op(Model(), (torch.randn(2, 3, 4, 5),), tester_factory)
    
    def test_logsigmoid_f32_boundary_values(self, tester_factory: Callable) -> None:
        # Test with specific values spanning negative and positive ranges
        x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        self._test_op(Model(), (x,), tester_factory)
