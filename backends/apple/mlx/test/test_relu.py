"""Tests for ReLU operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class ReluModel(nn.Module):
    """Model that applies ReLU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


@register_test
class ReluTest(OpTestCase):
    name = "relu"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"relu_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["ReluTest"]:
        return [
            cls(shape=(2, 3, 4)),  # 3D tensor
            cls(shape=(10,)),  # 1D tensor
            cls(shape=(4, 8)),  # 2D tensor
            cls(shape=(2, 8, 16)),  # Common intermediate layer
            cls(shape=(1, 128, 64)),  # Activation in attention layers
        ]

    def create_model(self) -> nn.Module:
        return ReluModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Use range that includes negative, zero, and positive values
        x = torch.randn(self.shape) * 2 - 1  # Range roughly [-3, 3]
        return (x,)
