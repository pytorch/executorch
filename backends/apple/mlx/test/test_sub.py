"""Tests for subtract operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class SubModel(nn.Module):
    """Model that performs element-wise subtraction."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sub(x, y)


@register_test
class SubTest(OpTestCase):
    name = "sub"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        scalar_sub: bool = False,
    ):
        self.shape = shape
        self.scalar_sub = scalar_sub
        shape_str = "x".join(str(s) for s in shape)
        if scalar_sub:
            self.name = f"sub_{shape_str}_scalar"
        else:
            self.name = f"sub_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["SubTest"]:
        return [
            cls(shape=(2, 3, 4)),  # 3D tensor
            cls(shape=(10,)),  # 1D tensor
            cls(shape=(4, 8)),  # 2D tensor
            cls(shape=(2, 8, 16)),  # Common intermediate layer
            cls(shape=(1, 128, 128)),  # Attention score adjustment
        ]

    def create_model(self) -> nn.Module:
        return SubModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        if self.scalar_sub:
            y = torch.randn(())  # Scalar
        else:
            y = torch.randn(self.shape)
        return (x, y)
