# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Log2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.log2(x)

@operator_test
class TestLog2(OperatorTest):
    @dtype_test
    def test_log2_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = Log2Model().to(dtype)
        # Use positive values only for log2
        self._test_op(model, (torch.rand(10, 10).to(dtype) + 0.01,), tester_factory)
        
    def test_log2_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with positive values
        self._test_op(Log2Model(), (torch.rand(10, 10) + 0.01,), tester_factory)
        
    def test_log2_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(Log2Model(), (torch.rand(20) + 0.01,), tester_factory)
        
        # 2D tensor
        self._test_op(Log2Model(), (torch.rand(5, 10) + 0.01,), tester_factory)
        
        # 3D tensor
        self._test_op(Log2Model(), (torch.rand(3, 4, 5) + 0.01,), tester_factory)
        
        # 4D tensor
        self._test_op(Log2Model(), (torch.rand(2, 3, 4, 5) + 0.01,), tester_factory)
        
        # 5D tensor
        self._test_op(Log2Model(), (torch.rand(2, 2, 3, 4, 5) + 0.01,), tester_factory)
        
    def test_log2_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small positive values (close to zero)
        self._test_op(Log2Model(), (torch.rand(10, 10) * 0.01 + 1e-6,), tester_factory)
        
        # Values around 1 (log2(1) = 0)
        self._test_op(Log2Model(), (torch.rand(10, 10) * 0.2 + 0.9,), tester_factory)
        
        # Values around powers of 2
        self._test_op(Log2Model(), (torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]).reshape(6, 1),), tester_factory)
        
        # Medium values
        self._test_op(Log2Model(), (torch.rand(10, 10) * 10 + 0.01,), tester_factory)
        
        # Large values
        self._test_op(Log2Model(), (torch.rand(10, 10) * 1000 + 0.01,), tester_factory)
        
        # Very large values
        self._test_op(Log2Model(), (torch.rand(5, 5) * 1e10 + 0.01,), tester_factory)
        
    def test_log2_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Values very close to zero
        self._test_op(Log2Model(), (torch.rand(10, 10) * 1e-6 + 1e-10,), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 0.5, 0.25, 0.125])
        self._test_op(Log2Model(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), 1.0, 2.0])
        self._test_op(Log2Model(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, 2.0])
        self._test_op(Log2Model(), (x,), tester_factory)
        
        # Tensor with ones (log2(1) = 0)
        self._test_op(Log2Model(), (torch.ones(10, 10),), tester_factory)
        
        # Tensor with twos (log2(2) = 1)
        self._test_op(Log2Model(), (torch.ones(10, 10) * 2,), tester_factory)
        
    def test_log2_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(Log2Model(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(Log2Model(), (torch.tensor([2.0]),), tester_factory)
        self._test_op(Log2Model(), (torch.tensor([0.5]),), tester_factory)
