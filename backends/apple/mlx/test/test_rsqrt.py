"""Tests for rsqrt operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class RsqrtModel(nn.Module):
    """Model that computes reciprocal square root."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x)


@register_test
class RsqrtTest(OpTestCase):
    name = "rsqrt"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"rsqrt_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["RsqrtTest"]:
        return [
            cls(shape=(2, 3, 4)),  # 3D tensor
            cls(shape=(10,)),  # 1D tensor
            cls(shape=(4, 8)),  # 2D tensor
            cls(shape=(2, 8, 16)),  # Common intermediate layer
            cls(shape=(1, 64)),  # Layer norm denominator
        ]

    def create_model(self) -> nn.Module:
        return RsqrtModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Use positive values to avoid NaN/complex results
        x = torch.rand(self.shape) + 0.1  # Range [0.1, 1.1]
        return (x,)
