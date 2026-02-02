"""
Test cases for to.dtype (astype) operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class ToDtypeModel(nn.Module):
    """Model that converts tensor to a different dtype."""

    def __init__(self, target_dtype: torch.dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.target_dtype)


@register_test
class ToDtypeTest(OpTestCase):
    """Test case for to.dtype op."""

    name = "to_dtype"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        source_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.bfloat16,
    ):
        self.shape = shape
        self.source_dtype = source_dtype
        self.target_dtype = target_dtype
        shape_str = "x".join(str(s) for s in shape)
        src_str = str(source_dtype).replace("torch.", "")
        tgt_str = str(target_dtype).replace("torch.", "")
        self.name = f"to_dtype_{shape_str}_{src_str}_to_{tgt_str}"

    @classmethod
    def get_test_configs(cls) -> List["ToDtypeTest"]:
        """Return all test configurations to run."""
        return [
            # Float32 to BFloat16
            cls(
                shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.bfloat16
            ),
            cls(shape=(10,), source_dtype=torch.float32, target_dtype=torch.bfloat16),
            cls(
                shape=(1, 128), source_dtype=torch.float32, target_dtype=torch.bfloat16
            ),
            # BFloat16 to Float32
            cls(
                shape=(2, 3, 4), source_dtype=torch.bfloat16, target_dtype=torch.float32
            ),
            cls(
                shape=(4, 8, 16),
                source_dtype=torch.bfloat16,
                target_dtype=torch.float32,
            ),
            # Float32 to Float16
            cls(
                shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.float16
            ),
            # Float16 to Float32
            cls(
                shape=(2, 3, 4), source_dtype=torch.float16, target_dtype=torch.float32
            ),
            # Int32 conversions
            cls(shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.int32),
            cls(shape=(2, 3, 4), source_dtype=torch.int32, target_dtype=torch.float32),
            # Int64 conversions
            cls(shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.int64),
            cls(shape=(2, 3, 4), source_dtype=torch.int64, target_dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.source_dtype in (torch.int32, torch.int64):
            x = torch.randint(-100, 100, self.shape, dtype=self.source_dtype)
        else:
            x = torch.randn(self.shape, dtype=self.source_dtype)
        return (x,)

    def create_model(self) -> nn.Module:
        return ToDtypeModel(self.target_dtype)
