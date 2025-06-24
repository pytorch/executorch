# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class PermuteModel(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(self.dims)

@operator_test
class TestPermute(OperatorTest):
    @dtype_test
    def test_permute_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = PermuteModel(dims=[1, 0])
        self._test_op(model, (torch.rand(3, 4).to(dtype),), tester_factory)
        
    def test_permute_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Permute a 2D tensor from [3, 4] to [4, 3]
        self._test_op(PermuteModel(dims=[1, 0]), (torch.randn(3, 4),), tester_factory)
        
    def test_permute_3d(self, tester_factory: Callable) -> None:
        # Test permuting a 3D tensor
        
        # Permute from [2, 3, 4] to [4, 2, 3]
        self._test_op(PermuteModel(dims=[2, 0, 1]), (torch.randn(2, 3, 4),), tester_factory)
        
        # Permute from [2, 3, 4] to [3, 4, 2]
        self._test_op(PermuteModel(dims=[1, 2, 0]), (torch.randn(2, 3, 4),), tester_factory)
        
        # Permute from [2, 3, 4] to [2, 4, 3]
        self._test_op(PermuteModel(dims=[0, 2, 1]), (torch.randn(2, 3, 4),), tester_factory)
        
    def test_permute_4d(self, tester_factory: Callable) -> None:
        # Test permuting a 4D tensor
        
        # Permute from [2, 3, 4, 5] to [5, 4, 3, 2]
        self._test_op(PermuteModel(dims=[3, 2, 1, 0]), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # Permute from [2, 3, 4, 5] to [2, 4, 3, 5]
        self._test_op(PermuteModel(dims=[0, 2, 1, 3]), (torch.randn(2, 3, 4, 5),), tester_factory)
        
    def test_permute_identity(self, tester_factory: Callable) -> None:
        # Test identity permutation (no change)
        
        # 2D tensor
        self._test_op(PermuteModel(dims=[0, 1]), (torch.randn(3, 4),), tester_factory)
        
        # 3D tensor
        self._test_op(PermuteModel(dims=[0, 1, 2]), (torch.randn(2, 3, 4),), tester_factory)
        
    def test_permute_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # 1D tensor (no permutation possible)
        self._test_op(PermuteModel(dims=[0]), (torch.randn(5),), tester_factory)
        
        # 5D tensor
        self._test_op(PermuteModel(dims=[4, 3, 2, 1, 0]), (torch.randn(2, 3, 4, 5, 6),), tester_factory)
