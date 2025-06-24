# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AcoshModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.acosh(x)

@operator_test
class TestAcosh(OperatorTest):
    @dtype_test
    def test_acosh_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        # Input values must be >= 1 for acosh
        model = AcoshModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 10 + 1,), tester_factory)
        
    def test_acosh_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with values >= 1
        self._test_op(AcoshModel(), (torch.rand(10, 10) * 10 + 1,), tester_factory)
        
    def test_acosh_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(AcoshModel(), (torch.rand(20) * 10 + 1,), tester_factory)
        
        # 2D tensor
        self._test_op(AcoshModel(), (torch.rand(5, 10) * 10 + 1,), tester_factory)
        
        # 3D tensor
        self._test_op(AcoshModel(), (torch.rand(3, 4, 5) * 10 + 1,), tester_factory)
        
        # 4D tensor
        self._test_op(AcoshModel(), (torch.rand(2, 3, 4, 5) * 10 + 1,), tester_factory)
        
        # 5D tensor
        self._test_op(AcoshModel(), (torch.rand(2, 2, 3, 4, 5) * 10 + 1,), tester_factory)
        
    def test_acosh_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges (all >= 1)
        
        # Values close to 1
        self._test_op(AcoshModel(), (torch.rand(10, 10) * 0.1 + 1,), tester_factory)
        
        # Medium values
        self._test_op(AcoshModel(), (torch.rand(10, 10) * 9 + 1,), tester_factory)
        
        # Large values
        self._test_op(AcoshModel(), (torch.rand(10, 10) * 100 + 1,), tester_factory)
        
        # Very large values
        self._test_op(AcoshModel(), (torch.rand(10, 10) * 1000 + 1,), tester_factory)
        
        # Specific values
        self._test_op(AcoshModel(), (torch.tensor([1.0, 1.5, 2.0, 5.0, 10.0]).view(5, 1),), tester_factory)
        
    def test_acosh_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Exactly 1 (boundary case)
        self._test_op(AcoshModel(), (torch.tensor([1.0, 1.0001]).view(2, 1),), tester_factory)
        
        # Empty tensor
        self._test_op(AcoshModel(), (torch.rand(0, 10) * 10 + 1,), tester_factory)
        
        # Single-element tensor
        self._test_op(AcoshModel(), (torch.tensor([1.0]).view(1, 1),), tester_factory)
        self._test_op(AcoshModel(), (torch.tensor([2.0]).view(1, 1),), tester_factory)
        
        # Infinity
        self._test_op(AcoshModel(), (torch.tensor([float('inf')]).view(1, 1),), tester_factory)
