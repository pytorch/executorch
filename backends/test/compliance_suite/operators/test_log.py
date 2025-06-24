# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class LogModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.log(x)

@operator_test
class TestLog(OperatorTest):
    @dtype_test
    def test_log_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = LogModel().to(dtype)
        # Use positive values only for log
        self._test_op(model, (torch.rand(10, 10).to(dtype) + 0.01,), tester_factory)
        
    def test_log_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with positive values
        self._test_op(LogModel(), (torch.rand(10, 10) + 0.01,), tester_factory)
        
    def test_log_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(LogModel(), (torch.rand(20) + 0.01,), tester_factory)
        
        # 2D tensor
        self._test_op(LogModel(), (torch.rand(5, 10) + 0.01,), tester_factory)
        
        # 3D tensor
        self._test_op(LogModel(), (torch.rand(3, 4, 5) + 0.01,), tester_factory)
        
        # 4D tensor
        self._test_op(LogModel(), (torch.rand(2, 3, 4, 5) + 0.01,), tester_factory)
        
        # 5D tensor
        self._test_op(LogModel(), (torch.rand(2, 2, 3, 4, 5) + 0.01,), tester_factory)
        
    def test_log_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small positive values (close to zero)
        self._test_op(LogModel(), (torch.rand(10, 10) * 0.01 + 1e-6,), tester_factory)
        
        # Values around 1 (log(1) = 0)
        self._test_op(LogModel(), (torch.rand(10, 10) * 0.2 + 0.9,), tester_factory)
        
        # Medium values
        self._test_op(LogModel(), (torch.rand(10, 10) * 10 + 0.01,), tester_factory)
        
        # Large values
        self._test_op(LogModel(), (torch.rand(10, 10) * 1000 + 0.01,), tester_factory)
        
        # Very large values
        self._test_op(LogModel(), (torch.rand(5, 5) * 1e10 + 0.01,), tester_factory)
        
    def test_log_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Values very close to zero
        self._test_op(LogModel(), (torch.rand(10, 10) * 1e-6 + 1e-10,), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([1.0, 2.0, 10.0, 100.0, 0.1, 0.01])
        self._test_op(LogModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), 1.0, 2.0])
        self._test_op(LogModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, 2.0])
        self._test_op(LogModel(), (x,), tester_factory)
        
        # Tensor with ones (log(1) = 0)
        self._test_op(LogModel(), (torch.ones(10, 10),), tester_factory)
        
        # Tensor with e (log(e) = 1)
        self._test_op(LogModel(), (torch.ones(10, 10) * torch.e,), tester_factory)
        
    def test_log_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(LogModel(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(LogModel(), (torch.tensor([2.0]),), tester_factory)
        self._test_op(LogModel(), (torch.tensor([0.5]),), tester_factory)
