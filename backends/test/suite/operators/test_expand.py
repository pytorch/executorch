# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ExpandModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.expand(self.shape)

@operator_test
class TestExpand(OperatorTest):
    @dtype_test
    def test_expand_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = ExpandModel(shape=[3, 5])
        self._test_op(model, (torch.rand(1, 5).to(dtype),), tester_factory)
        
    def test_expand_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Expand from [1, 5] to [3, 5]
        self._test_op(ExpandModel(shape=[3, 5]), (torch.randn(1, 5),), tester_factory)
        
    def test_expand_dimensions(self, tester_factory: Callable) -> None:
        # Test expanding different dimensions
        
        # Expand first dimension
        self._test_op(ExpandModel(shape=[3, 5]), (torch.randn(1, 5),), tester_factory)
        
        # Expand multiple dimensions
        self._test_op(ExpandModel(shape=[3, 4]), (torch.randn(1, 1),), tester_factory)
        
        # Expand with adding a new dimension at the beginning
        self._test_op(ExpandModel(shape=[2, 1, 5]), (torch.randn(1, 5),), tester_factory)
        
        # Expand with adding a new dimension in the middle
        self._test_op(ExpandModel(shape=[3, 2, 5]), (torch.randn(3, 1, 5),), tester_factory)
        
        # Expand with adding a new dimension at the end
        self._test_op(ExpandModel(shape=[3, 5, 2]), (torch.randn(3, 5, 1),), tester_factory)
        
    def test_expand_keep_original_size(self, tester_factory: Callable) -> None:
        # Test with -1 to keep the original size
        
        # Keep the last dimension size
        self._test_op(ExpandModel(shape=[3, -1]), (torch.randn(1, 5),), tester_factory)
        
        # Keep the first dimension size
        self._test_op(ExpandModel(shape=[-1, 5]), (torch.randn(2, 1),), tester_factory)
        
        # Keep multiple dimension sizes
        self._test_op(ExpandModel(shape=[-1, 4, -1]), (torch.randn(2, 1, 3),), tester_factory)
        
    def test_expand_singleton_dimensions(self, tester_factory: Callable) -> None:
        # Test expanding singleton dimensions
        
        # Expand a scalar to a vector
        self._test_op(ExpandModel(shape=[5]), (torch.randn(1),), tester_factory)
        
        # Expand a scalar to a matrix
        self._test_op(ExpandModel(shape=[3, 4]), (torch.randn(1, 1),), tester_factory)
        
        # Expand a vector to a matrix by adding a dimension
        self._test_op(ExpandModel(shape=[3, 5]), (torch.randn(5),), tester_factory)
