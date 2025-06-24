# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class BmmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.bmm(x, y)

@operator_test
class TestBmm(OperatorTest):
    @dtype_test
    def test_bmm_dtype(self, dtype, tester_factory: Callable) -> None:
        # Test with different dtypes
        if dtype.is_complex:
            # Skip complex dtypes for now
            return
        
        model = BmmModel().to(dtype)
        # Create two batched matrices with compatible dimensions
        x = torch.rand(2, 3, 4).to(dtype)  # batch_size=2, n=3, m=4
        y = torch.rand(2, 4, 5).to(dtype)  # batch_size=2, m=4, p=5
        self._test_op(model, (x, y), tester_factory)
        
    def test_bmm_basic(self, tester_factory: Callable) -> None:
        x = torch.randn(2, 100, 800)
        y = torch.randn(2, 800, 1200)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
    def test_bmm_batch_sizes(self, tester_factory: Callable) -> None:
        # Test with different batch sizes
        
        # Batch size = 1
        x = torch.randn(1, 3, 4)
        y = torch.randn(1, 4, 5)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Batch size = 5
        x = torch.randn(5, 3, 4)
        y = torch.randn(5, 4, 5)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Batch size = 10
        x = torch.randn(10, 3, 4)
        y = torch.randn(10, 4, 5)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
    def test_bmm_matrix_sizes(self, tester_factory: Callable) -> None:
        # Test with different matrix sizes
        
        # Square matrices
        x = torch.randn(2, 4, 4)
        y = torch.randn(2, 4, 4)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Rectangular matrices (tall)
        x = torch.randn(2, 6, 3)
        y = torch.randn(2, 3, 4)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Rectangular matrices (wide)
        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Different shapes
        x = torch.randn(3, 2, 5)
        y = torch.randn(3, 5, 3)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
    def test_bmm_values(self, tester_factory: Callable) -> None:
        # Test with different value patterns
        
        # Identity matrices
        x = torch.randn(2, 3, 4)
        y = torch.zeros(2, 4, 4)
        for i in range(2):
            for j in range(4):
                y[i, j, j] = 1.0  # Set diagonal elements to 1
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Zero matrices
        x = torch.randn(2, 3, 4)
        y = torch.zeros(2, 4, 5)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with negative values
        x = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]], [[-5.0, -6.0], [-7.0, -8.0]]])
        y = torch.tensor([[[-9.0, -10.0], [-11.0, -12.0]], [[-13.0, -14.0], [-15.0, -16.0]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with mixed positive and negative values
        x = torch.tensor([[[1.0, -2.0], [-3.0, 4.0]], [[5.0, -6.0], [-7.0, 8.0]]])
        y = torch.tensor([[[-9.0, 10.0], [11.0, -12.0]], [[-13.0, 14.0], [15.0, -16.0]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with fractional values
        x = torch.tensor([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]])
        y = torch.tensor([[[8.5, 9.5], [10.5, 11.5]], [[12.5, 13.5], [14.5, 15.5]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Integer matrices
        x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        y = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
    def test_bmm_edge_cases(self, tester_factory: Callable) -> None:
        # Test edge cases
        
        # Matrices with all same values
        x = torch.ones(2, 3, 4)
        y = torch.ones(2, 4, 5)
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with very large values
        x = torch.tensor([[[1e5, 2e5], [3e5, 4e5]], [[5e5, 6e5], [7e5, 8e5]]])
        y = torch.tensor([[[9e5, 10e5], [11e5, 12e5]], [[13e5, 14e5], [15e5, 16e5]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with very small values
        x = torch.tensor([[[1e-5, 2e-5], [3e-5, 4e-5]], [[5e-5, 6e-5], [7e-5, 8e-5]]])
        y = torch.tensor([[[9e-5, 10e-5], [11e-5, 12e-5]], [[13e-5, 14e-5], [15e-5, 16e-5]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Matrices with mixed large and small values
        x = torch.tensor([[[1e5, 2e-5], [3e5, 4e-5]], [[5e-5, 6e5], [7e-5, 8e5]]])
        y = torch.tensor([[[9e-5, 10e5], [11e-5, 12e5]], [[13e5, 14e-5], [15e5, 16e-5]]])
        self._test_op(BmmModel(), (x, y), tester_factory)
        
        # Minimum size matrices (1x1)
        x = torch.randn(2, 1, 1)
        y = torch.randn(2, 1, 1)
        self._test_op(BmmModel(), (x, y), tester_factory)
