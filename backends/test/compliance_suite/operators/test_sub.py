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
    def forward(self, x, y):
        return x - y

class ModelAlpha(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        return torch.sub(x, y, alpha=self.alpha)

@operator_test
class Subtract(OperatorTest):
    @dtype_test
    def test_subtract_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(
            Model(),
            (
                (torch.rand(2, 10) * 100).to(dtype),
                (torch.rand(2, 10) * 100).to(dtype),
            ),
            tester_factory)
        
    def test_subtract_f32_bcast_first(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(), 
            (
                torch.randn(5),
                torch.randn(1, 5, 1, 5),
            ),
            tester_factory)
        
    def test_subtract_f32_bcast_second(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(), 
            (
                torch.randn(4, 4, 2, 7),
                torch.randn(2, 7),
            ),
            tester_factory)

    def test_subtract_f32_bcast_unary(self, tester_factory: Callable) -> None:
        self._test_op(
            Model(), 
            (
                torch.randn(5),
                torch.randn(1, 1, 5),
            ),
            tester_factory)
        
    def test_subtract_f32_alpha(self, tester_factory: Callable) -> None:
        self._test_op(
            ModelAlpha(alpha=2), 
            (
                torch.randn(1, 25),
                torch.randn(1, 25),
            ),
            tester_factory)
