"""
Test cases for aten.zeros and aten.ones operations.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class ZerosModel(nn.Module):
    """Model that creates a tensor filled with zeros."""

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a dummy input to make the model have an input
        return torch.zeros(self.shape, dtype=self.dtype)


class OnesModel(nn.Module):
    """Model that creates a tensor filled with ones."""

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a dummy input to make the model have an input
        return torch.ones(self.shape, dtype=self.dtype)


@register_test
class ZerosTest(OpTestCase):
    """Test case for aten.zeros op."""

    name = "zeros"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"zeros_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ZerosTest"]:
        """Return all test configurations to run."""
        return [
            # Basic shapes with float32
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
            cls(shape=(1, 128), dtype=torch.float32),
            cls(shape=(4, 8, 16), dtype=torch.float32),
            # BFloat16
            cls(shape=(2, 3, 4), dtype=torch.bfloat16),
            cls(shape=(8, 16), dtype=torch.bfloat16),
            # Float16
            cls(shape=(2, 3, 4), dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Dummy input - zeros doesn't actually use it
        x = torch.randn(1, dtype=torch.float32)
        return (x,)

    def create_model(self) -> nn.Module:
        return ZerosModel(self.shape, self.dtype)


@register_test
class OnesTest(OpTestCase):
    """Test case for aten.ones op."""

    name = "ones"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"ones_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["OnesTest"]:
        """Return all test configurations to run."""
        return [
            # Basic shapes with float32
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
            cls(shape=(1, 128), dtype=torch.float32),
            cls(shape=(4, 8, 16), dtype=torch.float32),
            # BFloat16
            cls(shape=(2, 3, 4), dtype=torch.bfloat16),
            cls(shape=(8, 16), dtype=torch.bfloat16),
            # Float16
            cls(shape=(2, 3, 4), dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Dummy input - ones doesn't actually use it
        x = torch.randn(1, dtype=torch.float32)
        return (x,)

    def create_model(self) -> nn.Module:
        return OnesModel(self.shape, self.dtype)
