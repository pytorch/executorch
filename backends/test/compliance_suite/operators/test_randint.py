# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Sequence

import sys
import torch
import math

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class RandintModel(torch.nn.Module):
    def __init__(
        self,
        low: int,
        high: int,
        size: List[int],
        dtype: torch.dtype = torch.int64,
    ):
        super().__init__()
        self.low = low
        self.high = high
        self.size = size
        self.dtype = dtype
        
    def forward(self, x):
        # Take a dummy input (expected to be zeros) to work around backend limitations
        # for graphs that take no runtime inputs (also a bug, but that's a separate issue).
        result = torch.randint(self.low, self.high, self.size, dtype=self.dtype)
        return result + x

@operator_test
class TestRandint(OperatorTest):
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
    
    @dtype_test
    def test_randint_dtype(self, dtype, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        low = 0
        high = 10
        model = RandintModel(low, high, size, dtype=dtype)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=dtype),),
            tester_factory=tester_factory,
            output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
        )
    
    def test_randint_increasing_rank(self, tester_factory: Callable) -> None:
        # Test with increasing rank
        base_size = [1024]
        low = 0
        high = 100
        
        for i in range(4):
            size = base_size.copy()
            for j in range(i):
                size.append(j + 1)
                
            model = RandintModel(low, high, size)
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size, dtype=torch.int64),),
                tester_factory=tester_factory,
                output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
            )
            
    def test_randint_smoke(self, tester_factory: Callable) -> None:
        # Basic smoke test
        size = [2, 3, 4, 128]
        low = 0
        high = 100
        model = RandintModel(low, high, size)
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.int64),),
            tester_factory=tester_factory,
            output_validator=lambda outputs: self._validate_uniform_int_distribution(outputs, low, high),
        )
        
    def test_randint_different_ranges(self, tester_factory: Callable) -> None:
        # Test with different ranges
        size = [1000]
        ranges = [
            (0, 2),    # Binary values
            (-5, 5),   # Negative to positive
            (100, 200), # Larger values
            (-100, -50), # Negative range
        ]
        
        for low, high in ranges:
            model = RandintModel(low, high, size)
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size, dtype=torch.int64),),
                tester_factory=tester_factory,
                output_validator=lambda outputs, l=low, h=high: self._validate_uniform_int_distribution(outputs, l, h),
            )
        
    def test_randint_large_tensor(self, tester_factory: Callable) -> None:
        # Test with a large tensor to get more stable statistics
        size = [10000]
        low = 0
        high = 10
        model = RandintModel(low, high, size)
        
        # For large tensors, we can use a more strict validation
        def validate_large_tensor(outputs):
            tensor = outputs[0]
            values = tensor.flatten()
            
            # Check range
            min_val = torch.min(values).item()
            max_val = torch.max(values).item()
            
            if min_val < low or max_val >= high:
                print(f"Values should be between {low} and {high-1}, but found min={min_val}, max={max_val}")
                return False
            
            # Check if all values in the range are present
            unique_values = set(values.tolist())
            expected_values = set(range(low, high))
            
            if not expected_values.issubset(unique_values):
                missing = expected_values - unique_values
                print(f"Missing values in the expected range: {missing}")
                return False
            
            # Check distribution
            # Count occurrences of each value
            counts = {}
            for val in values.tolist():
                counts[val] = counts.get(val, 0) + 1
            
            # Expected count for each value in a uniform distribution
            expected_count = tensor.numel() / (high - low)
            
            # Check if counts are within 10% of expected
            for val in range(low, high):
                actual_count = counts.get(val, 0)
                if abs(actual_count - expected_count) > 0.1 * expected_count:
                    print(f"Count for value {val} is {actual_count}, expected around {expected_count}")
                    return False
            
            return True
        
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.int64),),
            tester_factory=tester_factory,
            output_validator=validate_large_tensor,
        )
