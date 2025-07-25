# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SelectModel(torch.nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index
        
    def forward(self, x):
        return torch.select(x, dim=self.dim, index=self.index)

@operator_test
class TestSelect(OperatorTest):
    @dtype_test
    def test_select_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SelectModel(dim=0, index=0)
        self._test_op(model, (torch.rand(3, 4, 5).to(dtype),), tester_factory)
        
    def test_select_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Select the first slice along dimension 0 from a 3D tensor
        # Result will be a 2D tensor of shape [4, 5]
        self._test_op(SelectModel(dim=0, index=0), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_select_dimensions(self, tester_factory: Callable) -> None:
        # Test selecting along different dimensions
        
        # Select along dimension 0
        # From tensor of shape [3, 4, 5] -> Result will be of shape [4, 5]
        self._test_op(SelectModel(dim=0, index=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Select along dimension 1
        # From tensor of shape [3, 4, 5] -> Result will be of shape [3, 5]
        self._test_op(SelectModel(dim=1, index=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # Select along dimension 2
        # From tensor of shape [3, 4, 5] -> Result will be of shape [3, 4]
        self._test_op(SelectModel(dim=2, index=3), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_select_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # Select along the last dimension (dim=-1)
        # From tensor of shape [3, 4, 5] -> Result will be of shape [3, 4]
        self._test_op(SelectModel(dim=-1, index=2), (torch.randn(3, 4, 5),), tester_factory)
        
        # Select along the second-to-last dimension (dim=-2)
        # From tensor of shape [3, 4, 5] -> Result will be of shape [3, 5]
        self._test_op(SelectModel(dim=-2, index=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # Select along the third-to-last dimension (dim=-3)
        # From tensor of shape [3, 4, 5] -> Result will be of shape [4, 5]
        self._test_op(SelectModel(dim=-3, index=0), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_select_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # Select from a 2D tensor
        # From tensor of shape [3, 4] -> Result will be of shape [4]
        self._test_op(SelectModel(dim=0, index=1), (torch.randn(3, 4),), tester_factory)
        
        # Select from a 4D tensor
        # From tensor of shape [2, 3, 4, 5] -> Result will be of shape [2, 4, 5]
        self._test_op(SelectModel(dim=1, index=1), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_select_edge_indices(self, tester_factory: Callable) -> None:
        # Test with edge indices
        
        # Select the first element (index=0)
        self._test_op(SelectModel(dim=0, index=0), (torch.randn(3, 4, 5),), tester_factory)
        
        # Select the last element (index=size-1)
        self._test_op(SelectModel(dim=0, index=2), (torch.randn(3, 4, 5),), tester_factory)
