# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class CatModel(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        
    def forward(self, x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=self.dim)

@operator_test
class TestCat(OperatorTest):
    @dtype_test
    def test_cat_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = CatModel()
        self._test_op(
            model, 
            (
                torch.rand(2, 3).to(dtype),
                torch.rand(3, 3).to(dtype),
                torch.rand(4, 3).to(dtype),
            ), 
            tester_factory
        )
        
    def test_cat_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Concatenate 3 tensors along dimension 0
        # Tensors of shapes [2, 3], [3, 3], [4, 3] -> Result will be of shape [9, 3]
        self._test_op(
            CatModel(), 
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ), 
            tester_factory
        )
        
    def test_cat_dimensions(self, tester_factory: Callable) -> None:
        # Test concatenating along different dimensions
        
        # Concatenate along dimension 0 (default)
        # Tensors of shapes [2, 3], [3, 3], [4, 3] -> Result will be of shape [9, 3]
        self._test_op(
            CatModel(dim=0), 
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ), 
            tester_factory
        )
        
        # Concatenate along dimension 1
        # Tensors of shapes [3, 2], [3, 3], [3, 4] -> Result will be of shape [3, 9]
        self._test_op(
            CatModel(dim=1), 
            (
                torch.randn(3, 2),
                torch.randn(3, 3),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
        # Concatenate along dimension 2
        # Tensors of shapes [2, 3, 1], [2, 3, 2], [2, 3, 3] -> Result will be of shape [2, 3, 6]
        self._test_op(
            CatModel(dim=2), 
            (
                torch.randn(2, 3, 1),
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 3),
            ), 
            tester_factory
        )
        
    def test_cat_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # Concatenate along the last dimension (dim=-1)
        # For tensors of shape [3, 2], [3, 3], [3, 4], this is equivalent to dim=1
        # Result will be of shape [3, 9]
        self._test_op(
            CatModel(dim=-1), 
            (
                torch.randn(3, 2),
                torch.randn(3, 3),
                torch.randn(3, 4),
            ), 
            tester_factory
        )
        
        # Concatenate along the second-to-last dimension (dim=-2)
        # For tensors of shape [2, 3], [3, 3], [4, 3], this is equivalent to dim=0
        # Result will be of shape [9, 3]
        self._test_op(
            CatModel(dim=-2), 
            (
                torch.randn(2, 3),
                torch.randn(3, 3),
                torch.randn(4, 3),
            ), 
            tester_factory
        )
        
    def test_cat_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # Concatenate 1D tensors
        # Tensors of shapes [2], [3], [4] -> Result will be of shape [9]
        self._test_op(
            CatModel(), 
            (
                torch.randn(2),
                torch.randn(3),
                torch.randn(4),
            ), 
            tester_factory
        )
        
        # Concatenate 3D tensors along dimension 0
        # Tensors of shapes [1, 3, 4], [2, 3, 4], [3, 3, 4] -> Result will be of shape [6, 3, 4]
        self._test_op(
            CatModel(dim=0), 
            (
                torch.randn(1, 3, 4),
                torch.randn(2, 3, 4),
                torch.randn(3, 3, 4),
            ), 
            tester_factory
        )
        
        # Concatenate 3D tensors along dimension 1
        # Tensors of shapes [2, 1, 4], [2, 2, 4], [2, 3, 4] -> Result will be of shape [2, 6, 4]
        self._test_op(
            CatModel(dim=1), 
            (
                torch.randn(2, 1, 4),
                torch.randn(2, 2, 4),
                torch.randn(2, 3, 4),
            ), 
            tester_factory
        )
        
        # Concatenate 3D tensors along dimension 2
        # Tensors of shapes [2, 3, 1], [2, 3, 2], [2, 3, 3] -> Result will be of shape [2, 3, 6]
        self._test_op(
            CatModel(dim=2), 
            (
                torch.randn(2, 3, 1),
                torch.randn(2, 3, 2),
                torch.randn(2, 3, 3),
            ), 
            tester_factory
        )
        
    def test_cat_same_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of the same shape
        # Tensors of shapes [2, 3], [2, 3], [2, 3] -> Result will be of shape [6, 3]
        self._test_op(
            CatModel(), 
            (
                torch.randn(2, 3),
                torch.randn(2, 3),
                torch.randn(2, 3),
            ), 
            tester_factory
        )
