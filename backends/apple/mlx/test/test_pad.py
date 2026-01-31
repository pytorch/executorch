"""Tests for constant_pad_nd operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class PadModel(nn.Module):
    """Model that pads a tensor with a constant value."""

    def __init__(self, pad: Tuple[int, ...], value: float = 0.0):
        super().__init__()
        self.pad = pad
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, self.pad, mode="constant", value=self.value)


@register_test
class PadTest(OpTestCase):
    name = "pad"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        pad: Tuple[int, ...] = (1, 1, 1, 1),
        value: float = 0.0,
    ):
        self.shape = shape
        self.pad = pad
        self.value = value
        shape_str = "x".join(str(s) for s in shape)
        pad_str = "_".join(str(p) for p in pad)
        self.name = f"pad_{shape_str}_p{pad_str}_v{int(value)}"

    @classmethod
    def get_test_configs(cls) -> List["PadTest"]:
        return [
            cls(shape=(2, 3, 4), pad=(1, 1, 1, 1), value=0.0),  # 3D, pad last 2 dims
            cls(shape=(10,), pad=(2, 3), value=0.0),  # 1D, asymmetric padding
            cls(shape=(4, 8), pad=(1, 2), value=0.0),  # 2D, pad last dim only
            cls(shape=(2, 8, 16), pad=(1, 1, 2, 2), value=0.0),  # 3D, pad last 2 dims
            cls(shape=(1, 3, 32, 32), pad=(1, 1, 1, 1), value=0.0),  # 4D, image padding
            cls(shape=(2, 3, 4), pad=(1, 1, 1, 1), value=1.0),  # Non-zero padding
        ]

    def create_model(self) -> nn.Module:
        return PadModel(self.pad, self.value)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)
