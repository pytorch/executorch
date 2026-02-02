"""
Test cases for unary math operations.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Unary Op Models
# =============================================================================


class FloorModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class CeilModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x)


class SquareModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


class ExpModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class SinModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class CosModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class TanModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tan(x)


class AsinModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asin(x)


class AcosModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.acos(x)


class AtanModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)


class SinhModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x)


class CoshModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cosh(x)


class AsinhModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x)


class AcoshModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.acosh(x)


class AtanhModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atanh(x)


class Log2Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log2(x)


class Log10Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(x)


class Log1pModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)


class ErfModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class Expm1Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)


class RoundModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)


class ReciprocalModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(x)


class SqrtModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)


class AbsModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class NegModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x)


# =============================================================================
# Test Cases
# =============================================================================


@register_test
class FloorTest(OpTestCase):
    """Test case for aten.floor op."""

    name = "floor"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"floor_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["FloorTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype) * 10,)

    def create_model(self) -> nn.Module:
        return FloorModel()


@register_test
class CeilTest(OpTestCase):
    """Test case for aten.ceil op."""

    name = "ceil"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"ceil_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["CeilTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype) * 10,)

    def create_model(self) -> nn.Module:
        return CeilModel()


@register_test
class SquareTest(OpTestCase):
    """Test case for aten.square op."""

    name = "square"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"square_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["SquareTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return SquareModel()


@register_test
class ExpTest(OpTestCase):
    """Test case for aten.exp op."""

    name = "exp"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"exp_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ExpTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return ExpModel()


@register_test
class SinTest(OpTestCase):
    """Test case for aten.sin op."""

    name = "sin"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"sin_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["SinTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype) * 3.14159,)

    def create_model(self) -> nn.Module:
        return SinModel()


@register_test
class CosTest(OpTestCase):
    """Test case for aten.cos op."""

    name = "cos"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"cos_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["CosTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype) * 3.14159,)

    def create_model(self) -> nn.Module:
        return CosModel()


@register_test
class TanTest(OpTestCase):
    """Test case for aten.tan op."""

    name = "tan"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"tan_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["TanTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Avoid values near pi/2 where tan explodes
        return (torch.randn(self.shape, dtype=self.dtype) * 0.5,)

    def create_model(self) -> nn.Module:
        return TanModel()


@register_test
class AsinTest(OpTestCase):
    """Test case for aten.asin op."""

    name = "asin"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"asin_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AsinTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # asin requires inputs in [-1, 1]
        return (torch.rand(self.shape, dtype=self.dtype) * 2 - 1,)

    def create_model(self) -> nn.Module:
        return AsinModel()


@register_test
class AcosTest(OpTestCase):
    """Test case for aten.acos op."""

    name = "acos"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"acos_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AcosTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # acos requires inputs in [-1, 1]
        return (torch.rand(self.shape, dtype=self.dtype) * 2 - 1,)

    def create_model(self) -> nn.Module:
        return AcosModel()


@register_test
class AtanTest(OpTestCase):
    """Test case for aten.atan op."""

    name = "atan"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"atan_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AtanTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return AtanModel()


@register_test
class SinhTest(OpTestCase):
    """Test case for aten.sinh op."""

    name = "sinh"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"sinh_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["SinhTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return SinhModel()


@register_test
class CoshTest(OpTestCase):
    """Test case for aten.cosh op."""

    name = "cosh"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"cosh_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["CoshTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return CoshModel()


@register_test
class AsinhTest(OpTestCase):
    """Test case for aten.asinh op."""

    name = "asinh"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"asinh_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AsinhTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return AsinhModel()


@register_test
class AcoshTest(OpTestCase):
    """Test case for aten.acosh op."""

    name = "acosh"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"acosh_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AcoshTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # acosh requires inputs >= 1
        return (torch.rand(self.shape, dtype=self.dtype) + 1.0,)

    def create_model(self) -> nn.Module:
        return AcoshModel()


@register_test
class AtanhTest(OpTestCase):
    """Test case for aten.atanh op."""

    name = "atanh"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"atanh_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AtanhTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # atanh requires inputs in (-1, 1)
        return (torch.rand(self.shape, dtype=self.dtype) * 1.8 - 0.9,)

    def create_model(self) -> nn.Module:
        return AtanhModel()


@register_test
class Log2Test(OpTestCase):
    """Test case for aten.log2 op."""

    name = "log2"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"log2_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["Log2Test"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # log2 requires positive inputs
        return (torch.rand(self.shape, dtype=self.dtype) + 0.1,)

    def create_model(self) -> nn.Module:
        return Log2Model()


@register_test
class Log10Test(OpTestCase):
    """Test case for aten.log10 op."""

    name = "log10"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"log10_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["Log10Test"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # log10 requires positive inputs
        return (torch.rand(self.shape, dtype=self.dtype) + 0.1,)

    def create_model(self) -> nn.Module:
        return Log10Model()


@register_test
class Log1pTest(OpTestCase):
    """Test case for aten.log1p op."""

    name = "log1p"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"log1p_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["Log1pTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # log1p requires inputs > -1
        return (torch.rand(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return Log1pModel()


@register_test
class ErfTest(OpTestCase):
    """Test case for aten.erf op."""

    name = "erf"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"erf_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ErfTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return ErfModel()


@register_test
class Expm1Test(OpTestCase):
    """Test case for aten.expm1 op."""

    name = "expm1"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"expm1_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["Expm1Test"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return Expm1Model()


@register_test
class RoundTest(OpTestCase):
    """Test case for aten.round op.

    Note: round.decimals variant is not supported as it's not in Core ATen.
    Only testing round to integer.
    """

    name = "round"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"round_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["RoundTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype) * 10,)

    def create_model(self) -> nn.Module:
        return RoundModel()


@register_test
class ReciprocalTest(OpTestCase):
    """Test case for aten.reciprocal op."""

    name = "reciprocal"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"reciprocal_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ReciprocalTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Avoid values near zero
        return (torch.randn(self.shape, dtype=self.dtype) + 1.0,)

    def create_model(self) -> nn.Module:
        return ReciprocalModel()


@register_test
class SqrtTest(OpTestCase):
    """Test case for aten.sqrt op."""

    name = "sqrt"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"sqrt_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["SqrtTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # sqrt requires non-negative inputs
        return (torch.rand(self.shape, dtype=self.dtype) + 0.1,)

    def create_model(self) -> nn.Module:
        return SqrtModel()


@register_test
class AbsTest(OpTestCase):
    """Test case for aten.abs op."""

    name = "abs"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"abs_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AbsTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return AbsModel()


@register_test
class NegTest(OpTestCase):
    """Test case for aten.neg op."""

    name = "neg"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        self.name = f"neg_{shape_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["NegTest"]:
        return [
            cls(shape=(16,), dtype=torch.float32),
            cls(shape=(4, 4), dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return NegModel()
