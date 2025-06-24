# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class StackModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        
    def forward(self, x1, x2, x3):
        return torch.stack([x1, x2, x3], dim=self.dim)

@operator_test
class TestStack(OperatorTest):
    @dtype_test
    def test_stack_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = StackModel()
        self._test_op(
            model, 
            (
                torch.rand(3, 4).to(dtype),
                torch.rand(3, 4).to(dtype),
                torch.rand(3, 4).to(dtype),
            ), 
            tester_factory
        )
        
    def test_stack_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Stack 3 tensors of shape [3, 4] along dimension 0
        # Result will be of shape [3, 3, 4]
        self._test_op(
            StackModel(), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
    def test_stack_dimensions(self, tester_factory: Callable) -> None:
        # Test stacking along different dimensions
        
        # Stack along dimension 0 (default)
        # Result will be of shape [3, 3, 4]
        self._test_op(
            StackModel(dim=0), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
        # Stack along dimension 1
        # Result will be of shape [3, 3, 4]
        self._test_op(
            StackModel(dim=1), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
        # Stack along dimension 2
        # Result will be of shape [3, 4, 3]
        self._test_op(
            StackModel(dim=2), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
    def test_stack_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # Stack along the last dimension (dim=-1)
        # For tensors of shape [3, 4], this is equivalent to dim=2
        # Result will be of shape [3, 4, 3]
        self._test_op(
            StackModel(dim=-1), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
        # Stack along the second-to-last dimension (dim=-2)
        # For tensors of shape [3, 4], this is equivalent to dim=1
        # Result will be of shape [3, 3, 4]
        self._test_op(
            StackModel(dim=-2), 
            (
                torch.randn(3, 4),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
    def test_stack_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # Stack 1D tensors
        # Result will be of shape [3, 5]
        self._test_op(
            StackModel(), 
            (
                torch.randn(5),
                torch.randn(5),
                torch.randn(5),
            ), 
            tester_factory
        )
        
        # Stack 3D tensors
        # Result will be of shape [3, 2, 3, 4]
        self._test_op(
            StackModel(), 
            (
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
            ), 
            tester_factory
        )
