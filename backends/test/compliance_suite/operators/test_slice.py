# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SliceSimple(torch.nn.Module):
    def __init__(self, index=1):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

class SliceRange(torch.nn.Module):
    def forward(self, x):
        return x[1:3]

@operator_test
class TestSlice(OperatorTest):
    @dtype_test
    def test_slice_simple_dtype(self, dtype, tester_factory: Callable) -> None:
        self._test_op(SliceSimple().to(dtype), ((torch.rand(2, 3, 4)).to(dtype),), tester_factory)
    
    def test_slice_range(self, tester_factory: Callable) -> None:
        self._test_op(SliceRange(), ((torch.rand(2, 5, 4),),), tester_factory)
