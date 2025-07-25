# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Optional, Tuple, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class SqueezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.squeeze(x)

class SqueezeDimModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)

@operator_test
class TestSqueeze(OperatorTest):
    @dtype_test
    def test_squeeze_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = SqueezeModel()
        self._test_op(model, (torch.rand(1, 3, 1, 5).to(dtype),), tester_factory)
        
    def test_squeeze_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (remove all dimensions of size 1)
        self._test_op(SqueezeModel(), (torch.randn(1, 3, 1, 5),), tester_factory)
        
    def test_squeeze_specific_dimension(self, tester_factory: Callable) -> None:
        # Test squeezing specific dimensions
        
        # Squeeze first dimension (size 1)
        self._test_op(SqueezeDimModel(dim=0), (torch.randn(1, 3, 5),), tester_factory)
        
        # Squeeze middle dimension (size 1)
        self._test_op(SqueezeDimModel(dim=2), (torch.randn(3, 4, 1, 5),), tester_factory)
        
        # Squeeze last dimension (size 1)
        self._test_op(SqueezeDimModel(dim=-1), (torch.randn(3, 4, 5, 1),), tester_factory)
        
    def test_squeeze_no_effect(self, tester_factory: Callable) -> None:
        # Test cases where squeeze has no effect
        
        # Dimension specified is not size 1
        self._test_op(SqueezeDimModel(dim=1), (torch.randn(3, 4, 5),), tester_factory)
        
        # No dimensions of size 1
        self._test_op(SqueezeModel(), (torch.randn(3, 4, 5),), tester_factory)
        
    def test_squeeze_multiple_dims(self, tester_factory: Callable) -> None:
        # Test squeezing multiple dimensions of size 1
        
        # Multiple dimensions of size 1 (all removed with default parameters)
        self._test_op(SqueezeModel(), (torch.randn(1, 3, 1, 5, 1),), tester_factory)
        
        self._test_op(SqueezeDimModel(dim=(0, 1)), (torch.randn(1, 1, 1),), tester_factory)
