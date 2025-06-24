# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Sequence

import torch
import math

from executorch.backends.test.compliance_suite import (
    operator_test,
    OperatorTest,
)

class RandnLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Generate a tensor with the same shape and dtype as x, filled with random values
        # from a normal distribution with mean 0 and variance 1
        return torch.randn_like(x)

@operator_test
class TestRandnLike(OperatorTest):
    def _validate_normal_distribution(self, outputs: Sequence[torch.Tensor]) -> bool:
        """
        Validate that the tensor values follow a normal distribution with mean 0 and std 1.
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
        
        # For normal distribution with mean 0 and std 1:
        # - Mean should be close to 0
        # - Standard deviation should be close to 1
        # Using the same thresholds as in op_randn_test.cpp
        tolerance = 5.0 / math.sqrt(tensor.numel())
        
        # Check mean (should be close to 0)
        if abs(mean) > tolerance:
            print(f"Mean {mean} is too far from expected 0 (tolerance: {tolerance})")
            return False
        
        # Check standard deviation (should be close to 1)
        if std_dev <= 0:
            print("Standard deviation should be positive")
            return False
            
        if abs(std_dev - 1.0) > 0.1:
            print(f"Standard deviation {std_dev} is too far from expected 1.0")
            return False
        
        return True
    
    def test_randn_like_float16(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandnLikeModel()
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.ones(*size).to(torch.float16),),
            tester_factory=tester_factory,
            output_validator=self._validate_normal_distribution,
        )
    
    def test_randn_like_float32(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandnLikeModel()
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.ones(*size),),
            tester_factory=tester_factory,
            output_validator=self._validate_normal_distribution,
        )
    
    def test_randn_like_float64(self, tester_factory: Callable) -> None:
        # Use a large enough size to get stable statistics
        size = [1000]
        model = RandnLikeModel()
        
        # Use _test_op with our custom validator
        self._test_op(
            model=model,
            inputs=(torch.ones(*size).to(torch.float64),),
            tester_factory=tester_factory,
            output_validator=self._validate_normal_distribution,
        )
         
    def test_randn_like_increasing_rank(self, tester_factory: Callable) -> None:
        base_size = [1024]
        
        for i in range(4):
            size = base_size.copy()
            for j in range(i):
                size.append(j + 1)
                
            model = RandnLikeModel()
            self._test_op(
                model=model,
                inputs=(torch.ones(*size),),
                tester_factory=tester_factory,
                output_validator=self._validate_normal_distribution
            )
            
    def test_randn_like_smoke(self, tester_factory: Callable) -> None:
        size = [2, 3, 4, 128]
        model = RandnLikeModel()
        self._test_op(
            model=model,
            inputs=(torch.ones(*size),),
            tester_factory=tester_factory,
            output_validator=self._validate_normal_distribution
        )
        
    def test_randn_like_large_tensor(self, tester_factory: Callable) -> None:
        # Test with a large tensor to get more stable statistics
        size = [10000]
        model = RandnLikeModel()
        
        # Custom validator with tighter bounds for large tensors
        def validate_large_tensor(outputs):
            tensor = outputs[0]
                
            values = tensor.flatten()
            mean = torch.mean(values).item()
            var = torch.var(values, unbiased=False).item()
            std_dev = math.sqrt(var)
            
            # With a large sample size, we can expect tighter bounds
            if abs(mean) > 0.05:
                print(f"Mean {mean} is too far from expected 0 (tolerance: 0.05)")
                return False
                
            if abs(std_dev - 1.0) > 0.05:
                print(f"Standard deviation {std_dev} is too far from expected 1.0 (tolerance: 0.05)")
                return False
                
            return True
        
        self._test_op(
            model=model,
            inputs=(torch.ones(*size),),
            tester_factory=tester_factory,
            output_validator=validate_large_tensor
        )
