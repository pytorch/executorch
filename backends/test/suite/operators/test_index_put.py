# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Tuple

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class IndexPutModel(torch.nn.Module):
    def __init__(self, accumulate=False):
        super().__init__()
        self.accumulate = accumulate
        
    def forward(self, x, indices, values):
        # Clone the input to avoid modifying it in-place
        result = x.clone()
        # Apply index_put_ and return the modified tensor
        result.index_put_(indices, values, self.accumulate)
        return result

@operator_test
class TestIndexPut(OperatorTest):
    @dtype_test
    def test_index_put_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0]).to(dtype)
        model = IndexPutModel()
        self._test_op(model, ((torch.rand(5, 2) * 100).to(dtype), indices, values), tester_factory, use_random_test_inputs=False)
        
    def test_index_put_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(), (torch.randn(5, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
    def test_index_put_accumulate(self, tester_factory: Callable) -> None:
        # Test with accumulate=True and accumulate=False
        
        # Without accumulation (replace values)
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(accumulate=False), 
                     (torch.ones(5, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
        # With accumulation (add values)
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(accumulate=True), 
                     (torch.ones(5, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
    def test_index_put_shapes(self, tester_factory: Callable) -> None:
        # Test with different tensor shapes
        
        # 1D tensor
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5), indices, values), tester_factory, use_random_test_inputs=False)
        
        # 2D tensor
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
        # 3D tensor
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 3, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
        # 4D tensor
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), 
                  torch.tensor([0, 1]), torch.tensor([2, 3]))
        values = torch.tensor([10.0,])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 3, 2, 4), indices, values), tester_factory, use_random_test_inputs=False)

    def test_index_put_indices(self, tester_factory: Callable) -> None:
        # Test with different index patterns
        
        # Single index
        indices = (torch.tensor([2]),)
        values = torch.tensor([10.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 2), indices, values), tester_factory, use_random_test_inputs=False)
        
        # Multiple indices
        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([10.0, 20.0, 30.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 3), indices, values), tester_factory, use_random_test_inputs=False)
        
        # Repeated indices with accumulate=True (values add up)
        indices = (torch.tensor([1, 1, 3, 3]),)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self._test_op(IndexPutModel(accumulate=True), 
                     (torch.randn(5), indices, values), tester_factory, use_random_test_inputs=False)
        
    def test_index_put_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Put values in all positions
        indices = (torch.tensor([0, 1, 2, 3, 4]),)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        self._test_op(IndexPutModel(), 
                     (torch.randn(5, 5), indices, values), tester_factory, use_random_test_inputs=False)
        