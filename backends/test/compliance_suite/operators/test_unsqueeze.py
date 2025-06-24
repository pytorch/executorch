# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class UnsqueezeModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)

@operator_test
class TestUnsqueeze(OperatorTest):
    @dtype_test
    def test_unsqueeze_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = UnsqueezeModel(dim=1)
        self._test_op(model, (torch.rand(3, 5).to(dtype),), tester_factory)
        
    def test_unsqueeze_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Add dimension at position 1
        self._test_op(UnsqueezeModel(dim=1), (torch.randn(3, 5),), tester_factory)
        
    def test_unsqueeze_positions(self, tester_factory: Callable) -> None:
        # Test unsqueezing at different positions
        
        # Unsqueeze at the beginning (dim=0)
        self._test_op(UnsqueezeModel(dim=0), (torch.randn(3, 5),), tester_factory)
        
        # Unsqueeze in the middle (dim=1)
        self._test_op(UnsqueezeModel(dim=1), (torch.randn(3, 5),), tester_factory)
        
        # Unsqueeze at the end (dim=2)
        self._test_op(UnsqueezeModel(dim=2), (torch.randn(3, 5),), tester_factory)
        
    def test_unsqueeze_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # Unsqueeze at the end (dim=-1)
        self._test_op(UnsqueezeModel(dim=-1), (torch.randn(3, 5),), tester_factory)
        
        # Unsqueeze at the second-to-last position (dim=-2)
        self._test_op(UnsqueezeModel(dim=-2), (torch.randn(3, 5),), tester_factory)
        
        # Unsqueeze at the beginning (dim=-3)
        self._test_op(UnsqueezeModel(dim=-3), (torch.randn(3, 5),), tester_factory)
        
    def test_unsqueeze_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # 1D tensor
        self._test_op(UnsqueezeModel(dim=0), (torch.randn(5),), tester_factory)
        self._test_op(UnsqueezeModel(dim=1), (torch.randn(5),), tester_factory)
        
        # 3D tensor
        self._test_op(UnsqueezeModel(dim=0), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(UnsqueezeModel(dim=2), (torch.randn(3, 4, 5),), tester_factory)
        self._test_op(UnsqueezeModel(dim=3), (torch.randn(3, 4, 5),), tester_factory)
        