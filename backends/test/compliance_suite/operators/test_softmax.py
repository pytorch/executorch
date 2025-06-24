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
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=self.dim)

@operator_test
class TestSoftmax(OperatorTest):
    @dtype_test
    def test_softmax_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(Model(), ((torch.rand(2, 10) * 100).to(dtype),), tester_factory)
        
    def test_softmax_f32_dim_last(self, tester_factory: Callable) -> None:
        # Default dim is -1 (last dimension)
        self._test_op(Model(), (torch.randn(3, 4, 5),), tester_factory)

    def test_softmax_f32_dim_first(self, tester_factory: Callable) -> None:
        # Test with dim=0 (first dimension)
        self._test_op(Model(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_softmax_f32_dim_middle(self, tester_factory: Callable) -> None:
        # Test with dim=1 (middle dimension)
        self._test_op(Model(dim=1), (torch.randn(3, 4, 5),), tester_factory)
    
    def test_softmax_f32_1d_tensor(self, tester_factory: Callable) -> None:
        # Test with 1D tensor
        self._test_op(Model(), (torch.randn(10),), tester_factory)
        
    def test_softmax_f32_large_values(self, tester_factory: Callable) -> None:
        # Test with large values to check numerical stability
        x = torch.tensor([[1000.0, 0.0, -1000.0]])
        self._test_op(Model(), (x,), tester_factory)
