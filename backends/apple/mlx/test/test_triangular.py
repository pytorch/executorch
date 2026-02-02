"""
Test cases for triangular matrix operations (tril, triu).
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Triangular Op Models
# =============================================================================


class TrilModel(nn.Module):
    """Model that computes lower triangular part of a matrix."""

    def __init__(self, diagonal: int = 0):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x, diagonal=self.diagonal)


class TriuModel(nn.Module):
    """Model that computes upper triangular part of a matrix."""

    def __init__(self, diagonal: int = 0):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu(x, diagonal=self.diagonal)


# =============================================================================
# Tril Tests
# =============================================================================


@register_test
class TrilTest(OpTestCase):
    """Test case for aten.tril op."""

    name = "tril"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        diagonal: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.diagonal = diagonal
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        diag_str = f"d{diagonal}" if diagonal != 0 else ""
        self.name = f"tril_{shape_str}_{dtype_str}{diag_str}"

    @classmethod
    def get_test_configs(cls) -> List["TrilTest"]:
        return [
            # Square matrices with main diagonal
            cls(shape=(4, 4), diagonal=0, dtype=torch.float32),
            cls(shape=(8, 8), diagonal=0, dtype=torch.float32),
            # Non-square matrices
            cls(shape=(4, 6), diagonal=0, dtype=torch.float32),
            cls(shape=(6, 4), diagonal=0, dtype=torch.float32),
            # Different diagonals
            cls(shape=(4, 4), diagonal=1, dtype=torch.float32),  # Above main
            cls(shape=(4, 4), diagonal=-1, dtype=torch.float32),  # Below main
            cls(shape=(4, 4), diagonal=2, dtype=torch.float32),
            # Different dtypes
            cls(shape=(4, 4), diagonal=0, dtype=torch.bfloat16),
            # Batched matrices
            cls(shape=(2, 4, 4), diagonal=0, dtype=torch.float32),
            cls(shape=(2, 3, 4, 4), diagonal=0, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return TrilModel(diagonal=self.diagonal)


# =============================================================================
# Triu Tests
# =============================================================================


@register_test
class TriuTest(OpTestCase):
    """Test case for aten.triu op."""

    name = "triu"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        diagonal: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.diagonal = diagonal
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        diag_str = f"d{diagonal}" if diagonal != 0 else ""
        self.name = f"triu_{shape_str}_{dtype_str}{diag_str}"

    @classmethod
    def get_test_configs(cls) -> List["TriuTest"]:
        return [
            # Square matrices with main diagonal
            cls(shape=(4, 4), diagonal=0, dtype=torch.float32),
            cls(shape=(8, 8), diagonal=0, dtype=torch.float32),
            # Non-square matrices
            cls(shape=(4, 6), diagonal=0, dtype=torch.float32),
            cls(shape=(6, 4), diagonal=0, dtype=torch.float32),
            # Different diagonals
            cls(shape=(4, 4), diagonal=1, dtype=torch.float32),  # Above main
            cls(shape=(4, 4), diagonal=-1, dtype=torch.float32),  # Below main
            cls(shape=(4, 4), diagonal=2, dtype=torch.float32),
            # Different dtypes
            cls(shape=(4, 4), diagonal=0, dtype=torch.bfloat16),
            # Batched matrices
            cls(shape=(2, 4, 4), diagonal=0, dtype=torch.float32),
            cls(shape=(2, 3, 4, 4), diagonal=0, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return TriuModel(diagonal=self.diagonal)
