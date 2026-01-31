"""
Test cases for Tanh operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class TanhModel(nn.Module):
    """Model that applies tanh activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


@register_test
class TanhTest(OpTestCase):
    """Test case for tanh op."""

    name = "tanh"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"tanh_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["TanhTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(2, 3, 4)),  # 3D tensor
            cls(shape=(10,)),  # 1D tensor
            cls(shape=(4, 8)),  # 2D tensor
            cls(shape=(2, 8, 16)),  # Common intermediate layer size
            cls(shape=(1, 1, 128)),  # Single batch, long sequence
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Use a range of values including negative, zero, and positive
        # to test tanh behavior across different ranges
        # tanh outputs values in [-1, 1] range
        x = torch.randn(self.shape) * 3  # Scale to get values in [-6, 6] range
        return (x,)

    def create_model(self) -> nn.Module:
        return TanhModel()
