# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Sequence

import sys
import torch
import math

from executorch.backends.test.compliance_suite import (
    operator_test,
    OperatorTest,
)

class RandintLikeModel(torch.nn.Module):
    def __init__(
        self,
        low: int,
        high: int,
    ):
        super().__init__()
        self.low = low
        self.high = high
        
    def forward(self, x):
        # Generate a tensor with the same shape and dtype as x, filled with random integers
        # from a uniform distribution in the range [low, high)
        return torch.randint_like(x, self.low, self.high)

@operator_test
class TestRandintLike(OperatorTest):
    def _validate_uniform_int_distribution(self, outputs: Sequence[torch.Tensor], low: int, high: int) -> bool:
        """
        Validate that the tensor values follow a uniform distribution of integers in the range [low, high).
        Returns True if the distribution is valid, False otherwise.
        """
        tensor = outputs[0]
            
        # Flatten the tensor to calculate statistics
        values = tensor.flatten()
        
        # Check range (all values should be between low and high-1)
        min_val = torch.min(values).item()
        max_val = torch.max(values).item()
        
        if min_val < low or max_val >= high:
            print(f"Values should be between {low} and {high-1}, but found min={min_val}, max={max_val}", file=sys.stderr)
            return False
        
        # Calculate mean
        mean = torch.mean(values.float()).item()
        
        # For uniform distribution of integers from low to high-1:
        # - Mean should be close to (low + high - 1) / 2
        expected_mean = (low + high - 1) / 2
        
        # Tolerance depends on the range and number of elements
        # For a uniform distribution, the standard error of the mean is approximately
        # sqrt((high-low)^2/12) / sqrt(n)
        std_error = math.sqrt((high - low)**2 / 12) / math.sqrt(tensor.numel())
        tolerance = 5.0 * std_error  # 5 standard errors
        
        if abs(mean - expected_mean) > tolerance:
            print(f"Mean {mean} is too far from expected {expected_mean} (tolerance: {tolerance})", file=sys.stderr)
            return False
        
        return True
    
    def test_randint_like_int32(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        low = 0
        high = 10
        model = RandintLikeModel(low, high)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.int32),),
            tester_factory=tester_factory,
            output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
        )
    
    def test_randint_like_int64(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        low = 0
        high = 10
        model = RandintLikeModel(low, high)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.int64),),
            tester_factory=tester_factory,
            output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
        )
         
    def test_randint_like_increasing_rank(self, tester_factory: Callable) -> None:
        base_size = [1024]
        low = 0
        high = 100
        
        for i in range(4):
            size = base_size.copy()
            for j in range(i):
                size.append(j + 1)
                
            model = RandintLikeModel(low, high)
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size, dtype=torch.int64),),
                tester_factory=tester_factory,
                output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
            )
            
    def test_randint_like_smoke(self, tester_factory: Callable) -> None:
        size = [2, 3, 4, 128]
        low = 0
        high = 100
        model = RandintLikeModel(low, high)
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.int64),),
            tester_factory=tester_factory,
            output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
        )
        
    def test_randint_like_different_ranges(self, tester_factory: Callable) -> None:
        # Test with different ranges
        size = [1000]
        ranges = [
            (0, 2),    # Binary values
            (-5, 5),   # Negative to positive
            (100, 200), # Larger values
            (-100, -50), # Negative range
        ]
        
        for low, high in ranges:
            model = RandintLikeModel(low, high)
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size, dtype=torch.int64),),
                tester_factory=tester_factory,
                output_validator=lambda outputs, l=low, h=high: self._validate_uniform_int_distribution(outputs, l, h),
            )
    
