# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class RsqrtModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.rsqrt(x)

@operator_test
class TestRsqrt(OperatorTest):
    @dtype_test
    def test_rsqrt_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = RsqrtModel().to(dtype)
        # Use positive values only for rsqrt to avoid division by zero
        self._test_op(model, (torch.rand(10, 10).to(dtype) + 0.01,), tester_factory)
        
    def test_rsqrt_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Input: tensor with positive values
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 10 + 0.01,), tester_factory)
        
    def test_rsqrt_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        self._test_op(RsqrtModel(), (torch.rand(20) + 0.01,), tester_factory)
        
        # 2D tensor
        self._test_op(RsqrtModel(), (torch.rand(5, 10) + 0.01,), tester_factory)
        
        # 3D tensor
        self._test_op(RsqrtModel(), (torch.rand(3, 4, 5) + 0.01,), tester_factory)
        
        # 4D tensor
        self._test_op(RsqrtModel(), (torch.rand(2, 3, 4, 5) + 0.01,), tester_factory)
        
        # 5D tensor
        self._test_op(RsqrtModel(), (torch.rand(2, 2, 3, 4, 5) + 0.01,), tester_factory)
        
    def test_rsqrt_values(self, tester_factory: Callable) -> None:
        # Test with different value ranges
        
        # Small values (rsqrt of small values gives large results)
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 0.01 + 0.01,), tester_factory)
        
        # Values around 1 (rsqrt(1) = 1)
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 0.2 + 0.9,), tester_factory)
        
        # Perfect squares
        x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
        # Medium values
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 10 + 0.01,), tester_factory)
        
        # Large values (rsqrt of large values gives small results)
        self._test_op(RsqrtModel(), (torch.rand(10, 10) * 1000 + 0.01,), tester_factory)
        
        # Very large values
        self._test_op(RsqrtModel(), (torch.rand(5, 5) * 1e10 + 0.01,), tester_factory)
        
    def test_rsqrt_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Tensor with specific values
        x = torch.tensor([1.0, 2.0, 4.0, 0.25, 0.5, 0.01])
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
        # Tensor with infinity
        x = torch.tensor([float('inf'), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
        # Tensor with NaN
        x = torch.tensor([float('nan'), 1.0, 4.0])
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
        # Values very close to zero (but not zero)
        x = torch.tensor([1e-5, 1e-10, 1e-15])
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
        # Values where rsqrt(x) = 1/sqrt(x) has a simple result
        x = torch.tensor([1.0, 4.0, 9.0, 16.0])  # rsqrt gives [1.0, 0.5, 0.33..., 0.25]
        self._test_op(RsqrtModel(), (x,), tester_factory)
        
    def test_rsqrt_scalar(self, tester_factory: Callable) -> None:
        # Test with scalar input (1-element tensor)
        self._test_op(RsqrtModel(), (torch.tensor([1.0]),), tester_factory)
        self._test_op(RsqrtModel(), (torch.tensor([4.0]),), tester_factory)
        self._test_op(RsqrtModel(), (torch.tensor([0.25]),), tester_factory)
        self._test_op(RsqrtModel(), (torch.tensor([100.0]),), tester_factory)
