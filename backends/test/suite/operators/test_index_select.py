# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class IndexSelectModel(torch.nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, indices):
        return torch.index_select(x, self.dim, indices)

@operator_test
class TestIndexSelect(OperatorTest):
    @dtype_test
    def test_index_select_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        indices = torch.tensor([0, 2], dtype=torch.int64)
        model = IndexSelectModel(dim=0)
        self._test_op(model, ((torch.rand(5, 3) * 100).to(dtype), indices), tester_factory, use_random_test_inputs=False)
        
    def test_index_select_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
    def test_index_select_dimensions(self, tester_factory: Callable) -> None:
        # Test selecting along different dimensions
        
        # Select along dim 0
        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Select along dim 1
        indices = torch.tensor([0, 1], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=1), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Select along dim 2 in a 3D tensor
        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=2), (torch.randn(3, 4, 5), indices), tester_factory, use_random_test_inputs=False)
        
    def test_index_select_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        indices = torch.tensor([0, 1], dtype=torch.int64)
        
        # 1D tensor
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5), indices), tester_factory, use_random_test_inputs=False)
        
        # 2D tensor
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # 3D tensor
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3, 2), indices), tester_factory, use_random_test_inputs=False)
        
        # 4D tensor
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3, 2, 4), indices), tester_factory, use_random_test_inputs=False)
        
    def test_index_select_indices(self, tester_factory: Callable) -> None:
        # Test with different index patterns
        
        # Single index
        indices = torch.tensor([2], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Multiple indices
        indices = torch.tensor([0, 2, 4], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Repeated indices
        indices = torch.tensor([1, 1, 3, 3], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Reversed indices
        indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
    def test_index_select_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Select all indices
        indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Select from a dimension with size 1
        indices = torch.tensor([0], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.randn(1, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Select from a tensor with all zeros
        indices = torch.tensor([0, 1], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.zeros(5, 3), indices), tester_factory, use_random_test_inputs=False)
        
        # Select from a tensor with all ones
        indices = torch.tensor([0, 1], dtype=torch.int64)
        self._test_op(IndexSelectModel(dim=0), (torch.ones(5, 3), indices), tester_factory, use_random_test_inputs=False)
