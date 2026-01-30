"""
Test cases for Sigmoid operation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .test_utils import OpTestCase, register_test


class SigmoidModel(nn.Module):
    """Model that applies sigmoid activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


@register_test
class SigmoidTest(OpTestCase):
    """Test case for sigmoid op."""

    name = "sigmoid"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"sigmoid_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["SigmoidTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(2, 3, 4)),      # 3D tensor
            cls(shape=(10,)),          # 1D tensor
            cls(shape=(4, 8)),         # 2D tensor
            cls(shape=(2, 8, 16)),     # Common intermediate layer size
            cls(shape=(1, 1, 128)),    # Single batch, long sequence
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Use a range of values including negative, zero, and positive
        # to test sigmoid behavior across different ranges
        x = torch.randn(self.shape) * 2  # Scale to get values in [-4, 4] range
        return (x,)

    def create_model(self) -> nn.Module:
        return SigmoidModel()
