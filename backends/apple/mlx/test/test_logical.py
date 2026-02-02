"""
Test cases for logical and comparison operations.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Comparison Op Models
# =============================================================================


class LessModel(nn.Module):
    """Model that computes element-wise a < b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a < b


class LessEqualModel(nn.Module):
    """Model that computes element-wise a <= b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a <= b


class GreaterModel(nn.Module):
    """Model that computes element-wise a > b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a > b


class GreaterEqualModel(nn.Module):
    """Model that computes element-wise a >= b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a >= b


class EqualModel(nn.Module):
    """Model that computes element-wise a == b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a == b


class NotEqualModel(nn.Module):
    """Model that computes element-wise a != b."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a != b


# =============================================================================
# Logical Op Models
# =============================================================================


class LogicalNotModel(nn.Module):
    """Model that computes element-wise logical NOT."""

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(a)


class LogicalAndModel(nn.Module):
    """Model that computes element-wise logical AND."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(a, b)


class LogicalOrModel(nn.Module):
    """Model that computes element-wise logical OR."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_or(a, b)


# =============================================================================
# Comparison Tests
# =============================================================================


@register_test
class LessTest(OpTestCase):
    """Test case for aten.lt (less than) op."""

    name = "less"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"less_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["LessTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
            cls(shape=(4, 8), dtype=torch.bfloat16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return LessModel()


@register_test
class LessEqualTest(OpTestCase):
    """Test case for aten.le (less than or equal) op."""

    name = "less_equal"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"less_equal_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["LessEqualTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return LessEqualModel()


@register_test
class GreaterTest(OpTestCase):
    """Test case for aten.gt (greater than) op."""

    name = "greater"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"greater_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["GreaterTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return GreaterModel()


@register_test
class GreaterEqualTest(OpTestCase):
    """Test case for aten.ge (greater than or equal) op."""

    name = "greater_equal"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"greater_equal_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["GreaterEqualTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return GreaterEqualModel()


@register_test
class EqualTest(OpTestCase):
    """Test case for aten.eq (equal) op."""

    name = "equal"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"equal_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["EqualTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return EqualModel()


@register_test
class NotEqualTest(OpTestCase):
    """Test case for aten.ne (not equal) op."""

    name = "not_equal"

    def __init__(
        self, shape: Tuple[int, ...] = (2, 3, 4), dtype: torch.dtype = torch.float32
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"not_equal_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["NotEqualTest"]:
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randn(self.shape, dtype=self.dtype)
        b = torch.randn(self.shape, dtype=self.dtype)
        return (a, b)

    def create_model(self) -> nn.Module:
        return NotEqualModel()


# =============================================================================
# Logical Tests
# =============================================================================


@register_test
class LogicalNotTest(OpTestCase):
    """Test case for aten.logical_not op."""

    name = "logical_not"

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"logical_not_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["LogicalNotTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(10,)),
            cls(shape=(4, 8)),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randint(0, 2, self.shape, dtype=torch.bool)
        return (a,)

    def create_model(self) -> nn.Module:
        return LogicalNotModel()


@register_test
class LogicalAndTest(OpTestCase):
    """Test case for aten.logical_and op."""

    name = "logical_and"

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"logical_and_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["LogicalAndTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(10,)),
            cls(shape=(4, 8)),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randint(0, 2, self.shape, dtype=torch.bool)
        b = torch.randint(0, 2, self.shape, dtype=torch.bool)
        return (a, b)

    def create_model(self) -> nn.Module:
        return LogicalAndModel()


@register_test
class LogicalOrTest(OpTestCase):
    """Test case for aten.logical_or op."""

    name = "logical_or"

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"logical_or_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["LogicalOrTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(10,)),
            cls(shape=(4, 8)),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        a = torch.randint(0, 2, self.shape, dtype=torch.bool)
        b = torch.randint(0, 2, self.shape, dtype=torch.bool)
        return (a, b)

    def create_model(self) -> nn.Module:
        return LogicalOrModel()
