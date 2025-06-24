# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, List, Sequence

import torch
import math

from executorch.backends.test.compliance_suite import (
    operator_test,
    OperatorTest,
)

class RandModel(torch.nn.Module):
    def __init__(
        self,
        size: List[int],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.size = size
        self.dtype = dtype
        
    def forward(self, x):
        # Take a dummy input (expected to be zeros) to work around backend limitations
        # for graphs that take no runtime inputs (also a bug, but that's a separate issue).
        return torch.rand(self.size, dtype=self.dtype) + x

@operator_test
class TestRand(OperatorTest):
    def _validate_uniform_distribution(self, outputs: Sequence[torch.Tensor]) -> bool:
        """
        Validate that the tensor values follow a uniform distribution between 0 and 1.
        Returns True if the distribution is valid, False otherwise.
        """
        tensor = outputs[0]
            
        # Flatten the tensor to calculate statistics
        values = tensor.flatten()
        
        # Calculate mean
        mean = torch.mean(values).item()
        
        # Calculate standard deviation
        var = torch.var(values, unbiased=False).item()
        std_dev = math.sqrt(var)
        
        # For uniform distribution between 0 and 1:
        # - Mean should be close to 0.5
        # - Standard deviation should be close to 1/sqrt(12) ≈ 0.289
        # Using the same thresholds as in op_rand_test.cpp
        tolerance = 5.0 / math.sqrt(tensor.numel())
        
        # Check mean (should be close to 0.5)
        if abs(mean - 0.5) > tolerance:
            print(f"Mean {mean} is too far from expected 0.5 (tolerance: {tolerance})")
            return False
        
        # Check standard deviation (should be close to 1/sqrt(12))
        expected_std = 1.0 / math.sqrt(12)
        if std_dev <= 0:
            print("Standard deviation should be positive")
            return False
            
        if abs(std_dev - expected_std) > 0.1:
            print(f"Standard deviation {std_dev} is too far from expected {expected_std}")
            return False
        
        # Check range (all values should be between 0 and 1)
        min_val = torch.min(values).item()
        max_val = torch.max(values).item()
        
        if min_val < 0 or max_val > 1:
            print(f"Values should be between 0 and 1, but found min={min_val}, max={max_val}")
            return False
        
        return True
    
    def test_rand_float16(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandModel(size, dtype=torch.float16)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size).to(torch.float16),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
    
    def test_rand_float32(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandModel(size, dtype=torch.float32)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
    
    def test_rand_float64(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandModel(size, dtype=torch.float64)
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size).to(torch.float64),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution,
        )
        
    def test_rand_increasing_rank(self, tester_factory: Callable) -> None:
        # Test with increasing rank, similar to the Rank test in op_rand_test.cpp
        base_size = [1024]
        
        for i in range(4):
            size = base_size.copy()
            for j in range(i):
                size.append(j + 1)
                
            model = RandModel(size)
            self._test_op(
                model=model,
                inputs=(torch.zeros(*size),),
                tester_factory=tester_factory,
                output_validator=self._validate_uniform_distribution
            )
            
    def test_rand_smoke(self, tester_factory: Callable) -> None:
        # Similar to the SmokeTest in op_rand_test.cpp
        size = [2, 3, 4, 128]
        model = RandModel(size)
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size),),
            tester_factory=tester_factory,
            output_validator=self._validate_uniform_distribution
        )
        
    def test_rand_large_tensor(self, tester_factory: Callable) -> None:
        # Test with a large tensor to get more stable statistics
        size = [10000]
        model = RandModel(size)
        
        # Custom validator with tighter bounds for large tensors
        def validate_large_tensor(outputs):
            tensor = outputs[0]
                
            values = tensor.flatten()
            mean = torch.mean(values).item()
            var = torch.var(values, unbiased=False).item()
            std_dev = math.sqrt(var)
            expected_std = 1.0 / math.sqrt(12)
            
            # With a large sample size, we can expect tighter bounds
            if abs(mean - 0.5) > 0.02:
                print(f"Mean {mean} is too far from expected 0.5 (tolerance: 0.02)")
                return False
                
            if abs(std_dev - expected_std) > 0.02:
                print(f"Standard deviation {std_dev} is too far from expected {expected_std} (tolerance: 0.02)")
                return False
                
            # Check range (all values should be between 0 and 1)
            min_val = torch.min(values).item()
            max_val = torch.max(values).item()
            
            if min_val < 0 or max_val > 1:
                print(f"Values should be between 0 and 1, but found min={min_val}, max={max_val}")
                return False
                
            return True
        
        self._test_op(
            model=model,
            inputs=(torch.zeros(*size),),
            tester_factory=tester_factory,
            output_validator=validate_large_tensor
        )
