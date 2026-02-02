"""
Test cases for full_like, zeros_like, and ones_like operations.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Models
# =============================================================================


class ZerosLikeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class OnesLikeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class FullLikeModel(nn.Module):
    def __init__(self, fill_value: float):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.fill_value)


# =============================================================================
# Test Cases
# =============================================================================


@register_test
class ZerosLikeTest(OpTestCase):
    """Test case for aten.zeros_like op."""

    name = "zeros_like"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"zeros_like_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ZerosLikeTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return ZerosLikeModel()


@register_test
class OnesLikeTest(OpTestCase):
    """Test case for aten.ones_like op."""

    name = "ones_like"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"ones_like_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["OnesLikeTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return OnesLikeModel()


@register_test
class FullLikeTest(OpTestCase):
    """Test case for aten.full_like op."""

    name = "full_like"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        fill_value: float = 3.14,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"full_like_{shape_str}_v{fill_value}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["FullLikeTest"]:
        return [
            cls(shape=(16,), fill_value=3.14, dtype=torch.float32),
            cls(shape=(4, 4), fill_value=2.71, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=-1.0, dtype=torch.float32),
            cls(shape=(4, 4), fill_value=0.5, dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return FullLikeModel(fill_value=self.fill_value)
