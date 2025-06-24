# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SplitSizeModel(torch.nn.Module):
    def __init__(self, split_size: int, dim: int = 0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim
        
    def forward(self, x):
        return torch.split(x, self.split_size, dim=self.dim)

class SplitSectionsModel(torch.nn.Module):
    def __init__(self, sections: List[int], dim: int = 0):
        super().__init__()
        self.sections = sections
        self.dim = dim
        
    def forward(self, x):
        return torch.split(x, self.sections, dim=self.dim)

@operator_test
class TestSplit(OperatorTest):
    @dtype_test
    def test_split_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SplitSizeModel(split_size=2)
        self._test_op(model, (torch.rand(6, 4).to(dtype),), tester_factory)
        
    def test_split_size_basic(self, tester_factory: Callable) -> None:
        # Basic test with split_size
        # Split a 6x4 tensor into chunks of size 2 along dimension 0
        self._test_op(SplitSizeModel(split_size=2), (torch.randn(6, 4),), tester_factory)
        
    def test_split_size_dimensions(self, tester_factory: Callable) -> None:
        # Test splitting along different dimensions
        
        # Split along dimension 0
        self._test_op(SplitSizeModel(split_size=2, dim=0), (torch.randn(6, 4),), tester_factory)
        
        # Split along dimension 1
        self._test_op(SplitSizeModel(split_size=2, dim=1), (torch.randn(4, 6),), tester_factory)
        
        # Split along dimension 2
        self._test_op(SplitSizeModel(split_size=2, dim=2), (torch.randn(3, 4, 6),), tester_factory)
        
    def test_split_size_uneven(self, tester_factory: Callable) -> None:
        # Test with uneven splits (last chunk may be smaller)
        
        # Split a 7x4 tensor into chunks of size 3 along dimension 0
        # This will result in chunks of size [3, 3, 1]
        self._test_op(SplitSizeModel(split_size=3), (torch.randn(7, 4),), tester_factory)
        
        # Split a 4x7 tensor into chunks of size 3 along dimension 1
        # This will result in chunks of size [4x3, 4x3, 4x1]
        self._test_op(SplitSizeModel(split_size=3, dim=1), (torch.randn(4, 7),), tester_factory)
        
    def test_split_sections_basic(self, tester_factory: Callable) -> None:
        # Basic test with sections
        # Split a 6x4 tensor into sections [2, 3, 1] along dimension 0
        self._test_op(SplitSectionsModel(sections=[2, 3, 1]), (torch.randn(6, 4),), tester_factory)
        
    def test_split_sections_dimensions(self, tester_factory: Callable) -> None:
        # Test splitting into sections along different dimensions
        
        # Split along dimension 0
        self._test_op(SplitSectionsModel(sections=[2, 3, 1], dim=0), (torch.randn(6, 4),), tester_factory)
        
        # Split along dimension 1
        self._test_op(SplitSectionsModel(sections=[2, 3, 1], dim=1), (torch.randn(4, 6),), tester_factory)
        
        # Split along dimension 2
        self._test_op(SplitSectionsModel(sections=[2, 3, 1], dim=2), (torch.randn(3, 4, 6),), tester_factory)
        
    def test_split_negative_dim(self, tester_factory: Callable) -> None:
        # Test with negative dimensions (counting from the end)
        
        # Split along the last dimension (dim=-1)
        self._test_op(SplitSizeModel(split_size=2, dim=-1), (torch.randn(4, 6),), tester_factory)
        
        # Split along the second-to-last dimension (dim=-2)
        self._test_op(SplitSizeModel(split_size=2, dim=-2), (torch.randn(4, 6),), tester_factory)
        
        # Split into sections along the last dimension
        self._test_op(SplitSectionsModel(sections=[2, 3, 1], dim=-1), (torch.randn(4, 6),), tester_factory)
