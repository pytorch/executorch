"""
Test cases for aten.full operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class FullModel(nn.Module):
    """Model that creates a tensor filled with a constant value."""

    def __init__(self, shape: Tuple[int, ...], fill_value: float, dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a dummy input to make the model have an input
        # We use torch.full to create a tensor filled with fill_value
        return torch.full(self.shape, self.fill_value, dtype=self.dtype)


@register_test
class FullTest(OpTestCase):
    """Test case for aten.full op."""

    name = "full"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        fill_value: float = 1.5,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"full_{shape_str}_{fill_value}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["FullTest"]:
        """Return all test configurations to run."""
        return [
            # Basic shapes with float32
            cls(shape=(2, 3, 4), fill_value=1.5, dtype=torch.float32),
            cls(shape=(10,), fill_value=0.0, dtype=torch.float32),
            cls(shape=(1, 128), fill_value=-2.5, dtype=torch.float32),
            cls(shape=(4, 8, 16), fill_value=3.14159, dtype=torch.float32),
            # BFloat16
            cls(shape=(2, 3, 4), fill_value=1.0, dtype=torch.bfloat16),
            cls(shape=(8, 16), fill_value=-1.0, dtype=torch.bfloat16),
            # Float16
            cls(shape=(2, 3, 4), fill_value=2.0, dtype=torch.float16),
            # Integer fill values
            cls(shape=(2, 3, 4), fill_value=0.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=1.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=-1.0, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Dummy input - full doesn't actually use it
        x = torch.randn(1, dtype=torch.float32)
        return (x,)

    def create_model(self) -> nn.Module:
        return FullModel(self.shape, self.fill_value, self.dtype)
