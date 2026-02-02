"""
Test cases for reduction operations.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


# =============================================================================
# Reduction Op Models
# =============================================================================


class SumModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.sum(x)
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


class MeanModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.mean(x)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class VarModel(nn.Module):
    def __init__(self, dim=None, keepdim=False, correction=1):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.var(x, correction=self.correction)
        return torch.var(
            x, dim=self.dim, keepdim=self.keepdim, correction=self.correction
        )


class StdModel(nn.Module):
    def __init__(self, dim=None, keepdim=False, correction=1):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.std(x, correction=self.correction)
        return torch.std(
            x, dim=self.dim, keepdim=self.keepdim, correction=self.correction
        )


class ProdModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.prod(x)
        return torch.prod(x, dim=self.dim, keepdim=self.keepdim)


class AmaxModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.amax(x)
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)


class AminModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.amin(x)
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


class ArgmaxModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.argmax(x)
        return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)


class ArgminModel(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return torch.argmin(x)
        return torch.argmin(x, dim=self.dim, keepdim=self.keepdim)


# =============================================================================
# Test Cases
# =============================================================================


@register_test
class SumTest(OpTestCase):
    """Test case for aten.sum op."""

    name = "sum"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"sum_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["SumTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return SumModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class MeanTest(OpTestCase):
    """Test case for aten.mean op."""

    name = "mean"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"mean_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["MeanTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return MeanModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class VarTest(OpTestCase):
    """Test case for aten.var op."""

    name = "var"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        correction: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        corr_str = f"_corr{correction}"
        self.name = f"var_{shape_str}{dim_str}{kd_str}{corr_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["VarTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(
                shape=(4, 4), dim=-1, correction=0, dtype=torch.float32
            ),  # population var
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return VarModel(dim=self.dim, keepdim=self.keepdim, correction=self.correction)


@register_test
class StdTest(OpTestCase):
    """Test case for aten.std op."""

    name = "std"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        correction: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        corr_str = f"_corr{correction}"
        self.name = f"std_{shape_str}{dim_str}{kd_str}{corr_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["StdTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(
                shape=(4, 4), dim=-1, correction=0, dtype=torch.float32
            ),  # population std
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return StdModel(dim=self.dim, keepdim=self.keepdim, correction=self.correction)


@register_test
class ProdTest(OpTestCase):
    """Test case for aten.prod op."""

    name = "prod"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"prod_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ProdTest"]:
        return [
            cls(shape=(8,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Use values close to 1 to avoid overflow/underflow in product
        return (torch.randn(self.shape, dtype=self.dtype) * 0.5 + 1.0,)

    def create_model(self) -> nn.Module:
        return ProdModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class AmaxTest(OpTestCase):
    """Test case for aten.amax op."""

    name = "amax"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"amax_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AmaxTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return AmaxModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class AminTest(OpTestCase):
    """Test case for aten.amin op."""

    name = "amin"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"amin_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["AminTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return AminModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class ArgmaxTest(OpTestCase):
    """Test case for aten.argmax op."""

    name = "argmax"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"argmax_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ArgmaxTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all (flatten)
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return ArgmaxModel(dim=self.dim, keepdim=self.keepdim)


@register_test
class ArgminTest(OpTestCase):
    """Test case for aten.argmin op."""

    name = "argmin"

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 4),
        dim=None,
        keepdim: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        dim_str = f"_dim{dim}" if dim is not None else "_all"
        kd_str = "_kd" if keepdim else ""
        self.name = f"argmin_{shape_str}{dim_str}{kd_str}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["ArgminTest"]:
        return [
            cls(shape=(16,), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, dtype=torch.float32),
            cls(shape=(4, 4), dim=0, dtype=torch.float32),
            cls(shape=(4, 4), dim=-1, keepdim=True, dtype=torch.float32),
            cls(shape=(2, 3, 4), dim=1, dtype=torch.float32),
            cls(shape=(4, 4), dim=None, dtype=torch.float32),  # reduce all (flatten)
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return ArgminModel(dim=self.dim, keepdim=self.keepdim)
