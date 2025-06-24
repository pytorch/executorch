# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class TransposeModel(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)

@operator_test
class TestTranspose(OperatorTest):
    @dtype_test
    def test_transpose_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = TransposeModel(dim0=0, dim1=1)
        self._test_op(model, (torch.rand(3, 4).to(dtype),), tester_factory)
        
    def test_transpose_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Transpose a 2D tensor from [3, 4] to [4, 3]
        self._test_op(TransposeModel(dim0=0, dim1=1), (torch.randn(3, 4),), tester_factory)
        
    def test_transpose_3d(self, tester_factory: Callable) -> None:
        # Test transposing a 3D tensor
        
        # Transpose dimensions 0 and 1
        # From [2, 3, 4] to [3, 2, 4]
        self._test_op(TransposeModel(dim0=0, dim1=1), (torch.randn(2, 3, 4),), tester_factory)
        
        # Transpose dimensions 0 and 2
        # From [2, 3, 4] to [4, 3, 2]
        self._test_op(TransposeModel(dim0=0, dim1=2), (torch.randn(2, 3, 4),), tester_factory)
        
        # Transpose dimensions 1 and 2
        # From [2, 3, 4] to [2, 4, 3]
        self._test_op(TransposeModel(dim0=1, dim1=2), (torch.randn(2, 3, 4),), tester_factory)
        
    def test_transpose_4d(self, tester_factory: Callable) -> None:
        # Test transposing a 4D tensor
        
        # Transpose dimensions 0 and 3
        # From [2, 3, 4, 5] to [5, 3, 4, 2]
        self._test_op(TransposeModel(dim0=0, dim1=3), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Transpose dimensions 1 and 2
        # From [2, 3, 4, 5] to [2, 4, 3, 5]
        self._test_op(TransposeModel(dim0=1, dim1=2), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_transpose_identity(self, tester_factory: Callable) -> None:
        # Test identity transpose (same dimension, no change)
        
        # 2D tensor
        self._test_op(TransposeModel(dim0=0, dim1=0), (torch.randn(3, 4),), tester_factory)
        self._test_op(TransposeModel(dim0=1, dim1=1), (torch.randn(3, 4),), tester_factory)
        
        # 3D tensor
        self._test_op(TransposeModel(dim0=0, dim1=0), (torch.randn(2, 3, 4),), tester_factory)
        self._test_op(TransposeModel(dim0=1, dim1=1), (torch.randn(2, 3, 4),), tester_factory)
        self._test_op(TransposeModel(dim0=2, dim1=2), (torch.randn(2, 3, 4),), tester_factory)
        
    def test_transpose_negative_dims(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # 3D tensor
        # Transpose dimensions -3 and -1 (equivalent to 0 and 2)
        # From [2, 3, 4] to [4, 3, 2]
        self._test_op(TransposeModel(dim0=-3, dim1=-1), (torch.randn(2, 3, 4),), tester_factory)
        
        # Transpose dimensions -2 and -1 (equivalent to 1 and 2)
        # From [2, 3, 4] to [2, 4, 3]
        self._test_op(TransposeModel(dim0=-2, dim1=-1), (torch.randn(2, 3, 4),), tester_factory)
        
    def test_transpose_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # 2D tensor
        self._test_op(TransposeModel(dim0=0, dim1=1), (torch.randn(3, 4),), tester_factory)
        
        # 3D tensor
        self._test_op(TransposeModel(dim0=0, dim1=2), (torch.randn(2, 3, 4),), tester_factory)
        
        # 4D tensor
        self._test_op(TransposeModel(dim0=1, dim1=3), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(TransposeModel(dim0=0, dim1=4), (torch.randn(2, 3, 4, 5, 6),), tester_factory)
