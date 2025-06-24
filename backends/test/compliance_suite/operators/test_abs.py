# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AbsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.abs(x)

@operator_test
class TestAbs(OperatorTest):
    @dtype_test
    def test_abs_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = AbsModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), tester_factory)
        
    def test_abs_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with positive and negative values
        self._test_op(AbsModel(), (torch.randn(10, 10),), tester_factory)
        
    def test_abs_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(AbsModel(), (torch.randn(20),), tester_factory)
        
        # 2D tensor
        self._test_op(AbsModel(), (torch.randn(5, 10),), tester_factory)
        
        # 3D tensor
        self._test_op(AbsModel(), (torch.randn(3, 4, 5),), tester_factory)
        
        # 4D tensor
        self._test_op(AbsModel(), (torch.randn(2, 3, 4, 5),), tester_factory)
        
        # 5D tensor
        self._test_op(AbsModel(), (torch.randn(2, 2, 3, 4, 5),), tester_factory)
        
    def test_abs_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 0.01,), tester_factory)
        
        # Large values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 1000,), tester_factory)
        
        # Mixed positive and negative values
        self._test_op(AbsModel(), (torch.randn(10, 10) * 10,), tester_factory)
        
        # All positive values
        self._test_op(AbsModel(), (torch.rand(10, 10) * 10,), tester_factory)
        
        # All negative values
        self._test_op(AbsModel(), (torch.rand(10, 10) * -10,), tester_factory)
        
        # Values close to zero
        self._test_op(AbsModel(), (torch.randn(10, 10) * 1e-5,), tester_factory)
        
    def test_abs_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor
        self._test_op(AbsModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), float('-inf'), 1.0, -1.0])
        self._test_op(AbsModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, -1.0])
        self._test_op(AbsModel(), (x,), tester_factory)
        
    def test_abs_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(AbsModel(), (torch.tensor([-5.0]),), tester_factory)
        self._test_op(AbsModel(), (torch.tensor([5.0]),), tester_factory)
        self._test_op(AbsModel(), (torch.tensor([0.0]),), tester_factory)
