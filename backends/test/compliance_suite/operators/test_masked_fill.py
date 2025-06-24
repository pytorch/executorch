# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class MaskedFillModel(torch.nn.Module):
    def __init__(self, value: Union[float, int]):
        super().__init__()
        self.value = value
        
    def forward(self, x, mask):
        return x.masked_fill(mask, self.value)

@operator_test
class TestMaskedFill(OperatorTest):
    @dtype_test
    def test_masked_fill_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        model = MaskedFillModel(value=0.0)
        self._test_op(
            model, 
            (
                torch.rand(3, 4).to(dtype),
                torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]]),
            ), 
            tester_factory
        )
        
    def test_masked_fill_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        # Fill with 0.0 where mask is True
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(3, 4),
                torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]]),
            ), 
            tester_factory
        )
        
    def test_masked_fill_different_values(self, tester_factory: Callable) -> None:
        # Test with different fill values
        
        # Fill with a positive value
        self._test_op(
            MaskedFillModel(value=5.0), 
            (
                torch.randn(3, 4),
                torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]]),
            ), 
            tester_factory
        )
        
        # Fill with a negative value
        self._test_op(
            MaskedFillModel(value=-5.0), 
            (
                torch.randn(3, 4),
                torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]]),
            ), 
            tester_factory
        )
        
        # Fill with an integer value
        self._test_op(
            MaskedFillModel(value=1), 
            (
                torch.randn(3, 4),
                torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]]),
            ), 
            tester_factory
        )
        
    def test_masked_fill_different_shapes(self, tester_factory: Callable) -> None:
        # Test with tensors of different shapes
        
        # 1D tensor
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(5),
                torch.tensor([True, False, True, False, True]),
            ), 
            tester_factory
        )
        
        # 3D tensor
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(2, 3, 4),
                torch.tensor([
                    [[True, False, True, False], [False, True, False, True], [True, True, False, False]],
                    [[False, False, True, True], [True, False, True, False], [False, True, False, True]]
                ]),
            ), 
            tester_factory
        )
        
    def test_masked_fill_all_true(self, tester_factory: Callable) -> None:
        # Test with all mask values set to True
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(3, 4),
                torch.ones(3, 4, dtype=torch.bool),
            ), 
            tester_factory
        )
        
    def test_masked_fill_all_false(self, tester_factory: Callable) -> None:
        # Test with all mask values set to False
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(3, 4),
                torch.zeros(3, 4, dtype=torch.bool),
            ), 
            tester_factory
        )
        
    def test_masked_fill_broadcast(self, tester_factory: Callable) -> None:
        # Test with broadcasting mask
        # A 1D mask can be broadcast to a 2D tensor
        self._test_op(
            MaskedFillModel(value=0.0), 
            (
                torch.randn(3, 4),
                torch.tensor([True, False, True, False]),
            ), 
            tester_factory
        )
