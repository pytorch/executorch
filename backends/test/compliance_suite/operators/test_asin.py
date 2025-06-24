# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class AsinModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.asin(x)

@operator_test
class TestAsin(OperatorTest):
    @dtype_test
    def test_asin_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        # Input values must be in the range [-1, 1] for asin
        model = AsinModel().to(dtype)
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 1,), tester_factory)
        
    def test_asin_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with values in the range [-1, 1]
        self._test_op(AsinModel(), (torch.rand(10, 10) * 2 - 1,), tester_factory)
        
    def test_asin_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(AsinModel(), (torch.rand(20) * 2 - 1,), tester_factory)
        
        # 2D tensor
        self._test_op(AsinModel(), (torch.rand(5, 10) * 2 - 1,), tester_factory)
        
        # 3D tensor
        self._test_op(AsinModel(), (torch.rand(3, 4, 5) * 2 - 1,), tester_factory)
        
        # 4D tensor
        self._test_op(AsinModel(), (torch.rand(2, 3, 4, 5) * 2 - 1,), tester_factory)
        
        # 5D tensor
        self._test_op(AsinModel(), (torch.rand(2, 2, 3, 4, 5) * 2 - 1,), tester_factory)
        
    def test_asin_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges within [-1, 1]
        
        # Values close to -1
        self._test_op(AsinModel(), (torch.rand(10, 10) * 0.1 - 1,), tester_factory)
        
        # Values close to 0
        self._test_op(AsinModel(), (torch.rand(10, 10) * 0.2 - 0.1,), tester_factory)
        
        # Values close to 1
        self._test_op(AsinModel(), (torch.rand(10, 10) * 0.1 + 0.9,), tester_factory)
        
        # Full range [-1, 1]
        self._test_op(AsinModel(), (torch.rand(10, 10) * 2 - 1,), tester_factory)
        
        # Specific values
        self._test_op(AsinModel(), (torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).view(5, 1),), tester_factory)
        
    def test_asin_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Exactly -1 and 1
        self._test_op(AsinModel(), (torch.tensor([-1.0, 1.0]).view(2, 1),), tester_factory)
        
        # Single-element tensor
        self._test_op(AsinModel(), (torch.tensor([-0.5]).view(1, 1),), tester_factory)
        self._test_op(AsinModel(), (torch.tensor([0.5]).view(1, 1),), tester_factory)
