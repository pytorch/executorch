"""
Test cases for binary math operations and reduction ops.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Binary Op Models
# =============================================================================


class Atan2Model(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.atan2(a, b)


class LogaddexpModel(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(a, b)


class FloorDivideModel(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(a, b)


class PowerModel(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a, b)


class PowerScalarModel(nn.Module):
    """Model for pow with scalar exponent (Tensor_Scalar variant)."""

    def __init__(self, exponent: float):
        super().__init__()
        self.exponent = exponent

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.pow(a, self.exponent)


# =============================================================================
# Reduction Op Models
# =============================================================================


class LogsumexpModel(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x, dim=self.dim, keepdim=self.keepdim)


# =============================================================================
# Binary Op Tests
# =============================================================================


@register_test
class Atan2Test(OpTestCase):
    """Test case for aten.atan2 op."""

    name = "atan2"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"atan2_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["Atan2Test"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(self.shape, dtype=self.dtype),
            torch.randn(self.shape, dtype=self.dtype),
        )

    def create_model(self) -> nn.Module:
        return Atan2Model()


@register_test
class LogaddexpTest(OpTestCase):
    """Test case for aten.logaddexp op."""

    name = "logaddexp"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"logaddexp_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["LogaddexpTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(self.shape, dtype=self.dtype),
            torch.randn(self.shape, dtype=self.dtype),
        )

    def create_model(self) -> nn.Module:
        return LogaddexpModel()


@register_test
class FloorDivideTest(OpTestCase):
    """Test case for aten.floor_divide op."""

    name = "floor_divide"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"floor_divide_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["FloorDivideTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(self.shape, dtype=self.dtype) * 10,
            torch.randn(self.shape, dtype=self.dtype).abs() + 1,  # Avoid div by zero
        )

    def create_model(self) -> nn.Module:
        return FloorDivideModel()


@register_test
class PowerTest(OpTestCase):
    """Test case for aten.pow op (Tensor_Tensor variant)."""

    name = "power"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"power_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["PowerTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.rand(self.shape, dtype=self.dtype) + 0.5,  # Positive base
            torch.rand(self.shape, dtype=self.dtype) * 2,  # Small exponent
        )

    def create_model(self) -> nn.Module:
        return PowerModel()


@register_test
class PowerScalarTest(OpTestCase):
    """Test case for aten.pow op (Tensor_Scalar variant)."""

    name = "power_scalar"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        exponent: float = 2.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.exponent = exponent
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"power_scalar_{shape_str}_exp{exponent}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["PowerScalarTest"]:
        return [
            cls(shape=(16,), exponent=2.0, dtype=torch.float32),
            cls(shape=(4, 4), exponent=0.5, dtype=torch.float32),  # sqrt
            cls(shape=(4, 4), exponent=3.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), exponent=-1.0, dtype=torch.float32),  # reciprocal
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.rand(self.shape, dtype=self.dtype) + 0.5,)  # Positive base

    def create_model(self) -> nn.Module:
        return PowerScalarModel(self.exponent)


# =============================================================================
# Reduction Op Tests
# =============================================================================


@register_test
class LogsumexpTest(OpTestCase):
    """Test case for aten.logsumexp op."""

    name = "logsumexp"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim: int = -1,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        kd_str = "_kd" if keepdim else ""
        self.name = f"logsumexp_{shape_str}_dim{dim}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["LogsumexpTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return LogsumexpModel(dim=self.dim, keepdim=self.keepdim)
