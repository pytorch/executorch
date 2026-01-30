"""Tests for divide operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class DivModel(nn.Module):
    """Model that performs element-wise division."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.div(x, y)


@register_test
class DivTest(OpTestCase):
    name = "div"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        scalar_divisor: bool = False,
    ):
        self.shape = shape
        self.scalar_divisor = scalar_divisor
        shape_str = "x".join(str(s) for s in shape)
        if scalar_divisor:
            self.name = f"div_{shape_str}_scalar"
        else:
            self.name = f"div_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["DivTest"]:
        return [
            cls(shape=(2, 3, 4)),  # 3D tensor
            cls(shape=(10,)),  # 1D tensor
            cls(shape=(4, 8)),  # 2D tensor
            cls(shape=(2, 8, 16)),  # Common intermediate layer
            cls(shape=(1, 128, 64)),  # Attention head division
        ]

    def create_model(self) -> nn.Module:
        return DivModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape) + 2.0  # Shift to avoid zeros
        if self.scalar_divisor:
            y = torch.randn(()) + 2.0  # Scalar
        else:
            y = torch.randn(self.shape) + 2.0  # Avoid division by values near zero
        return (x, y)
