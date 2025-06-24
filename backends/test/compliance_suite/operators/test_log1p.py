# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class Log1pModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.log1p(x)

@operator_test
class TestLog1p(OperatorTest):
    @dtype_test
    def test_log1p_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = Log1pModel().to(dtype)
        # Use values greater than -1 for log1p
        self._test_op(model, (torch.rand(10, 10).to(dtype) * 2 - 0.5,), tester_factory)
        
    def test_log1p_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with values greater than -1
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 2 - 0.5,), tester_factory)
        
    def test_log1p_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(Log1pModel(), (torch.rand(20) * 2 - 0.5,), tester_factory)
        
        # 2D tensor
        self._test_op(Log1pModel(), (torch.rand(5, 10) * 2 - 0.5,), tester_factory)
        
        # 3D tensor
        self._test_op(Log1pModel(), (torch.rand(3, 4, 5) * 2 - 0.5,), tester_factory)
        
        # 4D tensor
        self._test_op(Log1pModel(), (torch.rand(2, 3, 4, 5) * 2 - 0.5,), tester_factory)
        
        # 5D tensor
        self._test_op(Log1pModel(), (torch.rand(2, 2, 3, 4, 5) * 2 - 0.5,), tester_factory)
        
    def test_log1p_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values close to zero
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.01,), tester_factory)
        
        # Values close to -1 (lower bound for log1p)
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.1 - 0.99,), tester_factory)
        
        # Values around 0 (log1p(0) = 0)
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 0.2 - 0.1,), tester_factory)
        
        # Medium positive values
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 10,), tester_factory)
        
        # Large positive values
        self._test_op(Log1pModel(), (torch.rand(10, 10) * 1000,), tester_factory)
        
        # Very large positive values
        self._test_op(Log1pModel(), (torch.rand(5, 5) * 1e10,), tester_factory)
        
    def test_log1p_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Zero tensor (log1p(0) = 0)
        self._test_op(Log1pModel(), (torch.zeros(10, 10),), tester_factory)
        
        # Tensor with specific values
        x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0])
        self._test_op(Log1pModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 0.0, 1.0])
        self._test_op(Log1pModel(), (x,), tester_factory)
        
        # Values very close to -1
        x = torch.tensor([-0.999, -0.9999, -0.99999])
        self._test_op(Log1pModel(), (x,), tester_factory)
        
        # Very small positive values (where log1p is more accurate than log(1+x))
        x = torch.tensor([1e-10, 1e-15, 1e-20])
        self._test_op(Log1pModel(), (x,), tester_factory)
        
    def test_log1p_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(Log1pModel(), (torch.tensor([0.0]),), tester_factory)
        self._test_op(Log1pModel(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(Log1pModel(), (torch.tensor([-0.5]),), tester_factory)
