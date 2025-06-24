# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ViewModel(torch.nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(self.shape)

@operator_test
class TestView(OperatorTest):
    @dtype_test
    def test_view_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = ViewModel(shape=[3, 5])
        self._test_op(model, (torch.rand(15).to(dtype),), tester_factory)
        
    def test_view_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # View from [15] to [3, 5]
        self._test_op(ViewModel(shape=[3, 5]), (torch.randn(15),), tester_factory)
        
    def test_view_dimensions(self, tester_factory: Callable) -> None:
        # Test viewing to different dimensions
        
        # View from 1D to 2D
        self._test_op(ViewModel(shape=[3, 5]), (torch.randn(15),), tester_factory)
        
        # View from 2D to 1D
        self._test_op(ViewModel(shape=[20]), (torch.randn(4, 5),), tester_factory)
        
        # View from 2D to 3D
        self._test_op(ViewModel(shape=[2, 2, 5]), (torch.randn(4, 5),), tester_factory)
        
        # View from 3D to 2D
        self._test_op(ViewModel(shape=[6, 4]), (torch.randn(3, 2, 4),), tester_factory)
        
    def test_view_inferred_dimension(self, tester_factory: Callable) -> None:
        # Test with inferred dimension (-1)
        
        # Infer the last dimension
        self._test_op(ViewModel(shape=[3, -1]), (torch.randn(15),), tester_factory)
        
        # Infer the first dimension
        self._test_op(ViewModel(shape=[-1, 5]), (torch.randn(15),), tester_factory)
        
        # Infer the middle dimension
        self._test_op(ViewModel(shape=[2, -1, 3]), (torch.randn(24),), tester_factory)
        