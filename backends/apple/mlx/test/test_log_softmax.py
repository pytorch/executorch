"""Tests for log_softmax operation."""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class LogSoftmaxModel(nn.Module):
    """Model that applies log_softmax."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=self.dim)


@register_test
class LogSoftmaxTest(OpTestCase):
    name = "log_softmax"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4), dim: int = -1):
        self.shape = shape
        self.dim = dim
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"log_softmax_{shape_str}_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["LogSoftmaxTest"]:
        return [
            cls(shape=(2, 3, 4), dim=-1),  # 3D, last dimension
            cls(shape=(10,), dim=0),  # 1D
            cls(shape=(4, 8), dim=1),  # 2D, along columns
            cls(shape=(2, 8, 16), dim=1),  # 3D, middle dimension
            cls(shape=(1, 128, 512), dim=-1),  # Classifier output
        ]

    def create_model(self) -> nn.Module:
        return LogSoftmaxModel(dim=self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)
