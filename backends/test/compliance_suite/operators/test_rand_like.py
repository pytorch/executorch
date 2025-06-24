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

class RandLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Generate a tensor with the same shape and dtype as x, filled with random values
        # from a uniform distribution in the range [0, 1)
        return torch.rand_like(x)

@operator_test
class TestRandLike(OperatorTest):
    def _validate_uniform_distribution(self, outputs: Sequence[torch.Tensor]) -> bool:
        """
        Validate that the tensor values follow a uniform distribution in the range [0, 1).
        Returns True if the distribution is valid, False otherwise.
        """
        tensor = outputs[0]
            
        # Flatten the tensor to calculate statistics
        values = tensor.flatten()
        
        # Check range (all values should be between 0 and 1)
        min_val = torch.min(values).item()
        max_val = torch.max(values).item()
        
        if min_val < 0.0 or max_val >= 1.0:
            print(f"Values should be between 0.0 and 1.0, but found min={min_val}, max={max_val}", file=sys.stderr)
            return False
        
        # Calculate mean
        mean = torch.mean(values).item()
        
        # For uniform distribution between 0 and 1:
        # - Mean should be close to 0.5
        expected_mean = 0.5
        
        # Tolerance depends on the number of elements
        # For a uniform distribution, the standard error of the mean is approximately
        # 1/sqrt(12*n)
        std_error = 1.0 / math.sqrt(12.0 * tensor.numel())
        tolerance = 5.0 * std_error  # 5 standard errors
        
        if abs(mean - expected_mean) > tolerance:
            print(f"Mean {mean} is too far from expected {expected_mean} (tolerance: {tolerance})", file=sys.stderr)
            return False
        
        # Calculate variance
        variance = torch.var(values).item()
        
        # For uniform distribution between 0 and 1:
        # - Variance should be close to 1/12 ≈ 0.0833
        expected_variance = 1.0 / 12.0
        
        # Tolerance for variance
        var_std_error = expected_variance * math.sqrt(2.0 / tensor.numel())
        var_tolerance = 5.0 * var_std_error
        
        if abs(variance - expected_variance) > var_tolerance:
            print(f"Variance {variance} is too far from expected {expected_variance} (tolerance: {var_tolerance})", file=sys.stderr)
            return False
        
        return True
    
    def test_rand_like_float32(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandLikeModel()
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.float32),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
    
    def test_rand_like_float64(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandLikeModel()
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.float64),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
         
    def test_rand_like_increasing_rank(self, tester_factory: Callable) -> None:
        base_size = [1024]
        
        for i in range(4):
            size = base_size.copy()
            for j in range(i):
                size.append(j + 1)
                
            model = RandLikeModel()
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size, dtype=torch.float32),),
                tester_factory=tester_factory,
                output_validator=self._validate_uniform_distribution,
            )
            
    def test_rand_like_smoke(self, tester_factory: Callable) -> None:
        size = [2, 3, 4, 128]
        model = RandLikeModel()
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.float32),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
        
    def test_rand_like_large_tensor(self, tester_factory: Callable) -> None:
        # Test with a large tensor to get more stable statistics
        size = [10000]
        model = RandLikeModel()
        
        # For large tensors, we can use a more detailed validation
        def validate_large_tensor(outputs):
            tensor = outputs[0]
            values = tensor.flatten()
            
            # Check range
            min_val = torch.min(values).item()
            max_val = torch.max(values).item()
            
            if min_val < 0.0 or max_val >= 1.0:
                print(f"Values should be between 0.0 and 1.0, but found min={min_val}, max={max_val}")
                return False
            
            # Check distribution by binning values
            num_bins = 10
            bin_counts = torch.histc(values, bins=num_bins, min=0.0, max=1.0)
            
            # Expected count for each bin in a uniform distribution
            expected_count = tensor.numel() / num_bins
            
            # Check if bin counts are within 5% of expected
            for i, count in enumerate(bin_counts):
                if abs(count.item() - expected_count) > 0.05 * expected_count:
                    bin_start = i / num_bins
                    bin_end = (i + 1) / num_bins
                    print(f"Count for bin [{bin_start:.1f}, {bin_end:.1f}) is {count.item()}, expected around {expected_count}")
                    return False
            
            # Calculate mean and variance
            mean = torch.mean(values).item()
            variance = torch.var(values).item()
            
            # Expected values for uniform distribution [0,1)
            expected_mean = 0.5
            expected_variance = 1.0 / 12.0
            
            # Tighter tolerances for large tensors
            mean_tolerance = 0.01
            var_tolerance = 0.01
            
            if abs(mean - expected_mean) > mean_tolerance:
                print(f"Mean {mean} is too far from expected {expected_mean}")
                return False
                
            if abs(variance - expected_variance) > var_tolerance:
                print(f"Variance {variance} is too far from expected {expected_variance}")
                return False
            
            return True
        
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.float32),),
            tester_factory=tester_factory,
            output_validator=validate_large_tensor,
        )
        
    def test_rand_like_half_precision(self, tester_factory: Callable) -> None:
        # Test with half precision (float16)
        size = [1000]
        model = RandLikeModel()
        
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size, dtype=torch.float16),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
