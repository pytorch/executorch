"""
Test cases for aten._native_batch_norm_legit_no_training operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class BatchNormModel(nn.Module):
    """Model that applies batch normalization (inference mode)."""

    def __init__(self, num_features: int, dtype: torch.dtype, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=affine, dtype=dtype)
        self.bn.eval()  # Set to eval mode for no_training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BatchNorm1dModel(nn.Module):
    """Model that applies 1D batch normalization (inference mode)."""

    def __init__(self, num_features: int, dtype: torch.dtype, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=affine, dtype=dtype)
        self.bn.eval()  # Set to eval mode for no_training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


@register_test
class BatchNorm2dTest(OpTestCase):
    """Test case for aten._native_batch_norm_legit_no_training op with 2D input."""

    name = "batch_norm_2d"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_features: int = 16,
        height: int = 8,
        width: int = 8,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.height = height
        self.width = width
        self.dtype = dtype
        dtype_str = str(dtype).replace("torch.", "")
        self.name = (
            f"batch_norm_2d_{batch_size}x{num_features}x{height}x{width}_{dtype_str}"
        )

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm2dTest"]:
        """Return all test configurations to run."""
        return [
            # Basic float32 tests
            cls(batch_size=1, num_features=16, height=8, width=8, dtype=torch.float32),
            cls(
                batch_size=2, num_features=32, height=16, width=16, dtype=torch.float32
            ),
            cls(batch_size=4, num_features=64, height=4, width=4, dtype=torch.float32),
            # BFloat16 tests
            cls(batch_size=2, num_features=16, height=8, width=8, dtype=torch.bfloat16),
            cls(batch_size=1, num_features=32, height=4, width=4, dtype=torch.bfloat16),
            # Float16 tests
            cls(batch_size=2, num_features=16, height=8, width=8, dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size,
            self.num_features,
            self.height,
            self.width,
            dtype=self.dtype,
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return BatchNormModel(self.num_features, self.dtype)


@register_test
class BatchNorm1dTest(OpTestCase):
    """Test case for aten._native_batch_norm_legit_no_training op with 1D input."""

    name = "batch_norm_1d"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_features: int = 16,
        seq_len: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.seq_len = seq_len
        self.dtype = dtype
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"batch_norm_1d_{batch_size}x{num_features}x{seq_len}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm1dTest"]:
        """Return all test configurations to run."""
        return [
            # Basic float32 tests
            cls(batch_size=1, num_features=16, seq_len=32, dtype=torch.float32),
            cls(batch_size=2, num_features=32, seq_len=64, dtype=torch.float32),
            # BFloat16 tests
            cls(batch_size=2, num_features=16, seq_len=32, dtype=torch.bfloat16),
            # Float16 tests
            cls(batch_size=2, num_features=16, seq_len=32, dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.num_features, self.seq_len, dtype=self.dtype
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return BatchNorm1dModel(self.num_features, self.dtype)


@register_test
class BatchNorm2dNoAffineTest(OpTestCase):
    """Test case for batch norm with affine=False (no weight/bias)."""

    name = "batch_norm_2d_no_affine"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_features: int = 16,
        height: int = 8,
        width: int = 8,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.height = height
        self.width = width
        self.dtype = dtype
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"batch_norm_2d_no_affine_{batch_size}x{num_features}x{height}x{width}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm2dNoAffineTest"]:
        """Return all test configurations to run."""
        return [
            cls(batch_size=1, num_features=16, height=8, width=8, dtype=torch.float32),
            cls(batch_size=2, num_features=32, height=4, width=4, dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size,
            self.num_features,
            self.height,
            self.width,
            dtype=self.dtype,
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return BatchNormModel(self.num_features, self.dtype, affine=False)


@register_test
class BatchNorm1dNoAffineTest(OpTestCase):
    """Test case for 1D batch norm with affine=False (no weight/bias)."""

    name = "batch_norm_1d_no_affine"
    rtol = 1e-3
    atol = 1e-3

    def __init__(
        self,
        batch_size: int = 2,
        num_features: int = 16,
        seq_len: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.seq_len = seq_len
        self.dtype = dtype
        dtype_str = str(dtype).replace("torch.", "")
        self.name = (
            f"batch_norm_1d_no_affine_{batch_size}x{num_features}x{seq_len}_{dtype_str}"
        )

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm1dNoAffineTest"]:
        """Return all test configurations to run."""
        return [
            cls(batch_size=1, num_features=16, seq_len=32, dtype=torch.float32),
            cls(batch_size=2, num_features=32, seq_len=64, dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.num_features, self.seq_len, dtype=self.dtype
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return BatchNorm1dModel(self.num_features, self.dtype, affine=False)
