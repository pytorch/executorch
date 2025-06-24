# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AtanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.atanh(x)

@operator_test
class TestAtanh(OperatorTest):
    @dtype_test
    def test_atanh_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        # Input must be in range [-1, 1] for atanh
        model = AtanhModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 1.98 - 0.99,), tester_factory)
        
    def test_atanh_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input must be in range [-1, 1] for atanh
        self._test_op(AtanhModel(), (torch.rand(10, 10) * 1.98 - 0.99,), tester_factory)
        
    def test_atanh_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        # Input must be in range [-1, 1] for atanh
        
        # 1D tensor
        self._test_op(AtanhModel(), (torch.rand(20) * 1.98 - 0.99,), tester_factory)
        
        # 2D tensor
        self._test_op(AtanhModel(), (torch.rand(5, 10) * 1.98 - 0.99,), tester_factory)
        
        # 3D tensor
        self._test_op(AtanhModel(), (torch.rand(3, 4, 5) * 1.98 - 0.99,), tester_factory)
        
        # 4D tensor
        self._test_op(AtanhModel(), (torch.rand(2, 3, 4, 5) * 1.98 - 0.99,), tester_factory)
        
        # 5D tensor
        self._test_op(AtanhModel(), (torch.rand(2, 2, 3, 4, 5) * 1.98 - 0.99,), tester_factory)
        
    def test_atanh_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges within [-1, 1]
        
        # Values close to -1
        self._test_op(AtanhModel(), (torch.rand(10, 10) * 0.1 - 0.95,), tester_factory)
        
        # Values close to 0
        self._test_op(AtanhModel(), (torch.rand(10, 10) * 0.2 - 0.1,), tester_factory)
        
        # Values close to 1
        self._test_op(AtanhModel(), (torch.rand(10, 10) * 0.1 + 0.85,), tester_factory)
        
        # Full range [-0.99, 0.99] (avoiding exact -1 and 1 which cause infinity)
        self._test_op(AtanhModel(), (torch.rand(10, 10) * 1.98 - 0.99,), tester_factory)
        
        # Specific values
        self._test_op(AtanhModel(), (torch.tensor([-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99]).view(7, 1),), tester_factory)
        
    def test_atanh_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero
        self._test_op(AtanhModel(), (torch.zeros(5, 5),), tester_factory)
        
        # Single-element tensor
        self._test_op(AtanhModel(), (torch.tensor([-0.5]).view(1, 1),), tester_factory)
        self._test_op(AtanhModel(), (torch.tensor([0.0]).view(1, 1),), tester_factory)
        self._test_op(AtanhModel(), (torch.tensor([0.5]).view(1, 1),), tester_factory)
        
        # Near boundary values (but not exactly at boundaries)
        self._test_op(AtanhModel(), (torch.tensor([-0.99, 0.99]).view(2, 1),), tester_factory)
        
        # NaN
        self._test_op(AtanhModel(), (torch.tensor([float('nan')]).view(1, 1),), tester_factory)
