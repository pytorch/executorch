#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Consolidated op tests for the MLX delegate.

This file contains all op tests organized by category. Each test class inherits
from OpTestCase and can be run via pytest or the run_all_tests.py script.

Usage:
    # Run all tests
    pytest test_ops.py

    # Run specific test class
    pytest test_ops.py::TestAdd

    # Run via run_all_tests
    python -m executorch.backends.apple.mlx.test.run_all_tests
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import custom ops for RoPE and KV cache tests
from executorch.backends.apple.mlx import (  # noqa: F401 - registers mlx ops  # noqa: F401 - registers mlx.rope
    custom_ops,
    ops,
)
from executorch.extension.llm.custom_ops import (  # noqa: F401 - registers llama.update_cache
    custom_ops as llama_ops,
)
from torch.export import Dim

from .test_utils import OpTestCase, register_test


# =============================================================================
# ARITHMETIC OPS
# =============================================================================


class AddTensorModel(nn.Module):
    """Add two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class AddScalarModel(nn.Module):
    """Add tensor and scalar."""

    def __init__(self, scalar: float = 1.0):
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scalar


@register_test
class AddTest(OpTestCase):
    """Test case for add op."""

    name = "add"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        scalar: Optional[float] = None,
    ):
        self.shape = shape
        self.scalar = scalar

        if scalar is not None:
            self.name = "add_scalar"
        else:
            self.name = "add"

    @classmethod
    def get_test_configs(cls) -> List["AddTest"]:
        return [
            cls(),  # tensor + tensor
            cls(scalar=2.5),  # tensor + scalar
        ]

    def create_model(self) -> nn.Module:
        if self.scalar is not None:
            return AddScalarModel(self.scalar)
        else:
            return AddTensorModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        if self.scalar is not None:
            return (x,)
        else:
            y = torch.randn(self.shape)
            return (x, y)


class SubModel(nn.Module):
    """Model that performs element-wise subtraction."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sub(x, y)


@register_test
class SubTest(OpTestCase):
    name = "sub"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        scalar_sub: bool = False,
    ):
        self.shape = shape
        self.scalar_sub = scalar_sub
        shape_str = "x".join(str(s) for s in shape)
        if scalar_sub:
            self.name = f"sub_{shape_str}_scalar"
        else:
            self.name = f"sub_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["SubTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(10,)),
            cls(shape=(4, 8)),
            cls(shape=(2, 8, 16)),
            cls(shape=(1, 128, 128)),
            cls(shape=(2, 3, 4), scalar_sub=True),
        ]

    def create_model(self) -> nn.Module:
        return SubModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        if self.scalar_sub:
            y = torch.randn(())
        else:
            y = torch.randn(self.shape)
        return (x, y)


class MulTensorModel(nn.Module):
    """Multiply two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class MulScalarModel(nn.Module):
    """Multiply tensor and scalar."""

    def __init__(self, scalar: float = 1.0):
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scalar


@register_test
class MulTest(OpTestCase):
    """Test case for mul op."""

    name = "mul"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        scalar: Optional[float] = None,
    ):
        self.shape = shape
        self.scalar = scalar

        if scalar is not None:
            self.name = "mul_scalar"
        else:
            self.name = "mul"

    @classmethod
    def get_test_configs(cls) -> List["MulTest"]:
        return [
            cls(),
            cls(scalar=2.5),
        ]

    def create_model(self) -> nn.Module:
        if self.scalar is not None:
            return MulScalarModel(self.scalar)
        else:
            return MulTensorModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        if self.scalar is not None:
            return (x,)
        else:
            y = torch.randn(self.shape)
            return (x, y)


class DivModel(nn.Module):
    """Model that performs element-wise division."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.div(x, y)


@register_test
class DivTest(OpTestCase):
    name = "div"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        scalar_divisor: bool = False,
    ):
        self.shape = shape
        self.scalar_divisor = scalar_divisor
        shape_str = "x".join(str(s) for s in shape)
        if scalar_divisor:
            self.name = f"div_{shape_str}_scalar"
        else:
            self.name = f"div_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["DivTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(10,)),
            cls(shape=(4, 8)),
            cls(shape=(2, 8, 16)),
            cls(shape=(1, 128, 64)),
            cls(shape=(2, 3, 4), scalar_divisor=True),
        ]

    def create_model(self) -> nn.Module:
        return DivModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape) + 2.0
        if self.scalar_divisor:
            y = torch.randn(()) + 2.0
        else:
            y = torch.randn(self.shape) + 2.0
        return (x, y)


# =============================================================================
# ACTIVATION OPS
# =============================================================================


class ClampModel(nn.Module):
    """Model that applies clamp with min and max."""

    def __init__(self, min_val: Optional[float], max_val: Optional[float]):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)


@register_test
class ClampTest(OpTestCase):
    """Test case for clamp op with various min/max combinations."""

    name = "clamp"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        self.shape = shape
        self.min_val = min_val
        self.max_val = max_val

        # Build descriptive name
        parts = ["clamp"]
        if min_val is not None:
            parts.append(f"min{min_val}")
        if max_val is not None:
            parts.append(f"max{max_val}")
        if min_val is None and max_val is None:
            parts.append("none")
        shape_str = "x".join(str(s) for s in shape)
        parts.append(shape_str)
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["ClampTest"]:
        return [
            # Only min specified
            cls(shape=(2, 3, 4), min_val=-0.5, max_val=None),
            # Only max specified
            cls(shape=(2, 3, 4), min_val=None, max_val=0.5),
            # Both min and max specified
            cls(shape=(2, 3, 4), min_val=-0.5, max_val=0.5),
            # Different shapes
            cls(shape=(10,), min_val=-1.0, max_val=1.0),
            cls(shape=(4, 8), min_val=0.0, max_val=None),  # ReLU-like
            cls(shape=(2, 8, 16), min_val=-0.25, max_val=0.75),
        ]

    def create_model(self) -> nn.Module:
        return ClampModel(self.min_val, self.max_val)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Create inputs with values that span beyond typical clamp range
        x = torch.randn(self.shape) * 2  # values roughly in [-4, 4]
        return (x,)


class GELUModel(nn.Module):
    """Simple model using GELU activation."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x)


@register_test
class GELUTest(OpTestCase):
    """Test case for GELU activation."""

    name = "gelu"

    def __init__(self, shape: Tuple[int, ...] = (2, 16, 64), approximate: str = "none"):
        self.shape = shape
        self.approximate = approximate
        self.name = f"gelu_{approximate}" if approximate != "none" else "gelu"

    @classmethod
    def get_test_configs(cls) -> List["GELUTest"]:
        return [
            cls(),
            cls(shape=(4, 32, 128)),
            cls(approximate="tanh"),
            cls(shape=(4, 32, 128), approximate="tanh"),
        ]

    def create_model(self) -> nn.Module:
        return GELUModel(approximate=self.approximate)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class SoftmaxModel(nn.Module):
    """Model that performs softmax along a specified dimension."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=self.dim)


@register_test
class SoftmaxTest(OpTestCase):
    """Test case for softmax op."""

    name = "softmax"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        dim: int = -1,
    ):
        self.shape = shape
        self.dim = dim
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"softmax_{shape_str}_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["SoftmaxTest"]:
        return [
            cls(shape=(2, 3, 4), dim=-1),
            cls(shape=(2, 3, 4), dim=1),
            cls(shape=(4, 8), dim=-1),
            cls(shape=(2, 4, 8, 16), dim=-1),
        ]

    def create_model(self) -> nn.Module:
        return SoftmaxModel(dim=self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class LogSoftmaxModel(nn.Module):
    """Model that applies log_softmax."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x, dim=self.dim)


@register_test
class LogSoftmaxTest(OpTestCase):
    name = "log_softmax"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4), dim: int = -1):
        self.shape = shape
        self.dim = dim
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"log_softmax_{shape_str}_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["LogSoftmaxTest"]:
        return [
            cls(shape=(2, 3, 4), dim=-1),
            cls(shape=(10,), dim=0),
            cls(shape=(4, 8), dim=1),
            cls(shape=(2, 8, 16), dim=1),
            cls(shape=(1, 128, 512), dim=-1),
        ]

    def create_model(self) -> nn.Module:
        return LogSoftmaxModel(dim=self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


# =============================================================================
# SHAPE OPS
# =============================================================================


class SqueezeModel(nn.Module):
    """Model that squeezes a tensor at specified dimensions."""

    def __init__(self, dims: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, dim=self.dims)


@register_test
class SqueezeTest(OpTestCase):
    """Test case for squeeze op."""

    name = "squeeze"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (1, 3, 1, 4),
        dims: Optional[Tuple[int, ...]] = (0, 2),
    ):
        self.shape = shape
        self.dims = dims
        shape_str = "x".join(str(s) for s in shape)
        if dims is None:
            dims_str = "all"
        elif len(dims) == 0:
            dims_str = "empty"
        else:
            dims_str = "_".join(str(d) for d in dims)
        self.name = f"squeeze_{shape_str}_dims{dims_str}"

    @classmethod
    def get_test_configs(cls) -> List["SqueezeTest"]:
        return [
            cls(shape=(1, 3, 1, 4), dims=(0, 2)),
            cls(shape=(1, 5, 1, 1), dims=(0,)),
            cls(shape=(3, 1, 4), dims=(1,)),
            cls(shape=(1, 1, 8), dims=(0, 1)),
            cls(shape=(2, 1, 3, 1), dims=(1, 3)),
            # Squeeze all singleton dims (no dims specified)
            cls(shape=(1, 3, 1, 4), dims=None),
            # Dims include non-size-1 axes (should be no-op for those axes)
            cls(shape=(1, 1, 1, 8198), dims=(0, 1, 2, 3)),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SqueezeModel(self.dims)


class UnsqueezeModel(nn.Module):
    """Model that unsqueezes a tensor at a given dimension."""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


@register_test
class UnsqueezeTest(OpTestCase):
    """Test case for unsqueeze op."""

    name = "unsqueeze"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 16, 64),
        dim: int = 0,
    ):
        self.shape = shape
        self.dim = dim
        self.name = f"unsqueeze_dim{dim}"

    @classmethod
    def get_test_configs(cls) -> List["UnsqueezeTest"]:
        return [
            cls(dim=0),
            cls(dim=1),
            cls(dim=-1),
        ]

    def create_model(self) -> nn.Module:
        return UnsqueezeModel(self.dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class PermuteModel(nn.Module):
    """Model that permutes tensor dimensions."""

    def __init__(self, dims: Tuple[int, ...] = (0, 2, 1, 3)):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class TransposeModel(nn.Module):
    """Model that transposes two dimensions."""

    def __init__(self, dim0: int = 1, dim1: int = 2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


@register_test
class PermuteTest(OpTestCase):
    """Test case for permute and transpose ops."""

    name = "permute"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 8, 16, 64),
        variant: str = "permute",
        permute_dims: Tuple[int, ...] = (0, 2, 1, 3),
        transpose_dims: Tuple[int, int] = (1, 2),
    ):
        self.shape = shape
        self.variant = variant
        self.permute_dims = permute_dims
        self.transpose_dims = transpose_dims

        if variant == "transpose":
            self.name = "transpose"
        else:
            self.name = "permute"

    @classmethod
    def get_test_configs(cls) -> List["PermuteTest"]:
        return [
            cls(variant="permute", permute_dims=(0, 2, 1, 3)),
            cls(variant="transpose", transpose_dims=(1, 2)),
        ]

    def create_model(self) -> nn.Module:
        if self.variant == "transpose":
            return TransposeModel(self.transpose_dims[0], self.transpose_dims[1])
        else:
            return PermuteModel(self.permute_dims)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class NarrowModel(nn.Module):
    """Model that narrows a tensor along a dimension."""

    def __init__(self, dim: int, start: int, length: int):
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.narrow(self.dim, self.start, self.length)


@register_test
class NarrowTest(OpTestCase):
    """Test case for tensor.narrow()."""

    name = "narrow"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 16, 8),
        dim: int = 1,
        start: int = 2,
        length: int = 8,
    ):
        self.shape = shape
        self.dim = dim
        self.start = start
        self.length = length
        self.name = f"narrow_dim{dim}_start{start}_len{length}"

    @classmethod
    def get_test_configs(cls) -> List["NarrowTest"]:
        return [
            cls(shape=(4, 16, 8), dim=1, start=2, length=8),
            cls(shape=(8, 8), dim=0, start=1, length=4),
            cls(shape=(2, 32, 4), dim=1, start=0, length=16),
        ]

    def create_model(self) -> nn.Module:
        return NarrowModel(self.dim, self.start, self.length)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class SelectModel(nn.Module):
    """Model that selects a single index along a dimension.

    torch.select(input, dim, index) returns input[..., index, ...] where
    the indexing happens at dimension `dim`. The selected dimension is removed.
    Maps to aten.select_copy.int -> MLX take(array, index, axis).
    """

    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.select(x, self.dim, self.index)


@register_test
class SelectTest(OpTestCase):
    """Test case for torch.select (aten.select_copy.int -> TakeNode)."""

    name = "select"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 8, 16),
        dim: int = 1,
        index: int = 3,
    ):
        self.shape = shape
        self.dim = dim
        self.index = index
        self.name = f"select_dim{dim}_idx{index}"

    @classmethod
    def get_test_configs(cls) -> List["SelectTest"]:
        return [
            cls(shape=(4, 8, 16), dim=0, index=2),
            cls(shape=(4, 8, 16), dim=1, index=3),
            cls(shape=(4, 8, 16), dim=2, index=0),
            cls(shape=(4, 8, 16), dim=-1, index=5),
            cls(shape=(2, 3), dim=0, index=1),
        ]

    def create_model(self) -> nn.Module:
        return SelectModel(self.dim, self.index)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class SliceModel(nn.Module):
    """Model that slices a tensor along dimension 1."""

    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start : self.stop]


class SliceDim0Model(nn.Module):
    """Model that slices a tensor along dimension 0."""

    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.start : self.stop]


@register_test
class SliceTest(OpTestCase):
    """Test case for tensor slicing."""

    name = "slice"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        shape: Tuple[int, ...] = (4, 16, 8),
        dim: int = 1,
        start: int = 2,
        stop: int = 10,
    ):
        self.shape = shape
        self.dim = dim
        self.start = start
        self.stop = stop
        self.name = f"slice_dim{dim}_{start}to{stop}"

    @classmethod
    def get_test_configs(cls) -> List["SliceTest"]:
        return [
            cls(shape=(4, 16, 8), dim=1, start=2, stop=10),
            cls(shape=(8, 8), dim=0, start=1, stop=5),
            cls(shape=(2, 32, 4), dim=1, start=0, stop=16),
        ]

    def create_model(self) -> nn.Module:
        if self.dim == 0:
            return SliceDim0Model(self.start, self.stop)
        return SliceModel(self.start, self.stop)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


class RepeatModel(nn.Module):
    """Model that repeats a tensor along specified dimensions."""

    def __init__(self, repeats: Tuple[int, ...]):
        super().__init__()
        self.repeats = repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(*self.repeats)


@register_test
class RepeatTest(OpTestCase):
    """Test case for tensor.repeat()."""

    name = "repeat"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (2, 3, 4),
        repeats: Tuple[int, ...] = (2, 1, 3),
    ):
        self.input_shape = input_shape
        self.repeats = repeats
        repeat_str = "x".join(str(r) for r in repeats)
        self.name = f"repeat_{repeat_str}"

    @classmethod
    def get_test_configs(cls) -> List["RepeatTest"]:
        return [
            cls(input_shape=(2, 3), repeats=(2, 3)),
            cls(input_shape=(2, 3, 4), repeats=(1, 2, 1)),
            cls(input_shape=(4, 4), repeats=(3, 3)),
        ]

    def create_model(self) -> nn.Module:
        return RepeatModel(self.repeats)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape)
        return (x,)


# =============================================================================
# TENSOR CONSTRUCTION OPS
# =============================================================================


class CatNModel(nn.Module):
    """Model that concatenates N tensors along a dimension."""

    def __init__(self, dim: int = 0, n: int = 3):
        super().__init__()
        self.dim = dim
        self.n = n

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensors[: self.n], dim=self.dim)


@register_test
class CatTest(OpTestCase):
    name = "cat"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes: List[Tuple[int, ...]], dim: int = 0, tag: str = ""):
        self.shapes = shapes
        self.dim = dim
        self.name = f"cat_{tag}" if tag else "cat"

    @classmethod
    def get_test_configs(cls) -> List["CatTest"]:
        return [
            cls(shapes=[(2, 3), (4, 3), (1, 3)], dim=0, tag="2d_dim0"),
            cls(shapes=[(3, 2), (3, 4), (3, 1)], dim=1, tag="2d_dim1"),
            cls(shapes=[(2, 3, 4), (5, 3, 4), (3, 3, 4)], dim=0, tag="3d_dim0"),
            cls(shapes=[(3, 4), (2, 4)], dim=0, tag="two_tensors"),
            cls(shapes=[(3, 2, 4), (3, 5, 4), (3, 1, 4)], dim=-2, tag="neg_dim"),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.randn(s) for s in self.shapes)

    def create_model(self) -> nn.Module:
        return CatNModel(dim=self.dim, n=len(self.shapes))


class WhereModel(nn.Module):
    """Model that conditionally selects from x or y based on condition."""

    def forward(
        self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(condition, x, y)


@register_test
class WhereTest(OpTestCase):
    """Test case for where op."""

    name = "where"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape: Tuple[int, ...] = (2, 3, 4)):
        self.shape = shape
        shape_str = "x".join(str(s) for s in shape)
        self.name = f"where_{shape_str}"

    @classmethod
    def get_test_configs(cls) -> List["WhereTest"]:
        return [
            cls(shape=(2, 3, 4)),
            cls(shape=(4, 8)),
            cls(shape=(2, 8, 16, 16)),
            cls(shape=(1, 1, 128, 128)),
        ]

    def create_model(self) -> nn.Module:
        return WhereModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        condition = torch.rand(self.shape) > 0.5
        x = torch.randn(self.shape)
        y = torch.randn(self.shape)
        return (condition, x, y)


class PadModel(nn.Module):
    """Model that pads a tensor with a constant value."""

    def __init__(self, pad: Tuple[int, ...], value: float = 0.0):
        super().__init__()
        self.pad = pad
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, self.pad, mode="constant", value=self.value)


@register_test
class PadTest(OpTestCase):
    name = "pad"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        shape: Tuple[int, ...] = (2, 3, 4),
        pad: Tuple[int, ...] = (1, 1, 1, 1),
        value: float = 0.0,
    ):
        self.shape = shape
        self.pad = pad
        self.value = value
        shape_str = "x".join(str(s) for s in shape)
        pad_str = "_".join(str(p) for p in pad)
        self.name = f"pad_{shape_str}_p{pad_str}_v{int(value)}"

    @classmethod
    def get_test_configs(cls) -> List["PadTest"]:
        return [
            cls(shape=(2, 3, 4), pad=(1, 1, 1, 1), value=0.0),
            cls(shape=(10,), pad=(2, 3), value=0.0),
            cls(shape=(4, 8), pad=(1, 2), value=0.0),
            cls(shape=(2, 8, 16), pad=(1, 1, 2, 2), value=0.0),
            cls(shape=(1, 3, 32, 32), pad=(1, 1, 1, 1), value=0.0),
            cls(shape=(2, 3, 4), pad=(1, 1, 1, 1), value=1.0),
        ]

    def create_model(self) -> nn.Module:
        return PadModel(self.pad, self.value)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)


# =============================================================================
# NEURAL NETWORK LAYER OPS
# =============================================================================


class LinearModel(nn.Module):
    """Simple linear layer for testing."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@register_test
class LinearTest(OpTestCase):
    """Test case for nn.Linear."""

    name = "linear"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias

        if not bias:
            self.name = "linear_no_bias"
        else:
            self.name = "linear"

    @classmethod
    def get_test_configs(cls) -> List["LinearTest"]:
        return [
            cls(),
            cls(bias=False),
        ]

    def create_model(self) -> nn.Module:
        return LinearModel(self.in_features, self.out_features, bias=self.bias)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        return (x,)


class EmbeddingModel(nn.Module):
    """Simple embedding layer for testing."""

    def __init__(self, num_embeddings: int = 1000, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


@register_test
class EmbeddingTest(OpTestCase):
    """Test case for nn.Embedding."""

    name = "embedding"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.name = "embedding"

    @classmethod
    def get_test_configs(cls) -> List["EmbeddingTest"]:
        return [
            cls(),
            cls(num_embeddings=512, embedding_dim=128),
        ]

    def create_model(self) -> nn.Module:
        return EmbeddingModel(self.num_embeddings, self.embedding_dim)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randint(0, self.num_embeddings, (self.batch_size, self.seq_len))
        return (x,)


# =============================================================================
# POOLING OPS
# =============================================================================


class MaxPool1dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class MaxPool1dTest(OpTestCase):
    name = "max_pool1d"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 8,
        seq_len: int = 32,
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.batch_size = batch_size

        if tag:
            self.name = f"max_pool1d_{tag}"
        else:
            parts = ["max_pool1d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["MaxPool1dTest"]:
        return [
            # Fast path: kernel == stride
            cls(kernel_size=2, stride=2),
            # General path: overlapping windows with padding
            cls(kernel_size=3, stride=2, padding=1),
            # Fast path: larger kernel
            cls(kernel_size=4, stride=4, seq_len=64),
            # stride=None (defaults to kernel_size)
            cls(kernel_size=4, stride=None, seq_len=64, tag="stride_none"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, tag="c1"),
            # Global pooling: kernel == spatial size
            cls(kernel_size=32, stride=32, tag="global"),
            # Batch > 1
            cls(kernel_size=2, stride=2, batch_size=4, tag="batch4"),
            # Stride > kernel (gaps between windows)
            cls(kernel_size=2, stride=3, seq_len=32, tag="stride_gt_kernel"),
        ]

    def create_model(self) -> nn.Module:
        return MaxPool1dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.batch_size, self.in_channels, self.seq_len),)


class MaxPool2dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class MaxPool2dTest(OpTestCase):
    name = "max_pool2d"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 16,
        input_size: Tuple[int, int] = (32, 32),
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.input_size = input_size
        self.batch_size = batch_size

        if tag:
            self.name = f"max_pool2d_{tag}"
        else:
            parts = ["max_pool2d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            parts.append(f"{input_size[0]}x{input_size[1]}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["MaxPool2dTest"]:
        return [
            # Fast path: kernel == stride, evenly divisible
            cls(kernel_size=2, stride=2, input_size=(32, 32)),
            # General path: overlapping windows
            cls(kernel_size=3, stride=2, padding=1, input_size=(32, 32)),
            # Fast path: 4x4 pooling
            cls(kernel_size=4, stride=4, input_size=(64, 64)),
            # General path: stride != kernel, no padding
            cls(kernel_size=3, stride=1, input_size=(16, 16)),
            # Batch > 1
            cls(kernel_size=2, stride=2, input_size=(32, 32), batch_size=4),
            # stride=None (defaults to kernel_size)
            cls(kernel_size=2, stride=None, input_size=(32, 32), tag="stride_none"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, input_size=(16, 16), tag="c1"),
            # Global pooling
            cls(kernel_size=8, stride=8, input_size=(8, 8), tag="global"),
            # Non-square kernel/stride
            cls(
                kernel_size=(2, 3),
                stride=(2, 3),
                input_size=(16, 18),
                tag="nonsquare_fast",
            ),
            cls(
                kernel_size=(3, 2),
                stride=(1, 2),
                padding=(1, 0),
                input_size=(16, 16),
                tag="nonsquare_general",
            ),
            # Stride > kernel (gaps between windows)
            cls(kernel_size=2, stride=3, input_size=(16, 16), tag="stride_gt_kernel"),
            # Non-square input with square kernel
            cls(kernel_size=2, stride=2, input_size=(16, 32), tag="nonsquare_input"),
        ]

    def create_model(self) -> nn.Module:
        return MaxPool2dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(
                self.batch_size,
                self.in_channels,
                self.input_size[0],
                self.input_size[1],
            ),
        )


class MaxPool3dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class MaxPool3dTest(OpTestCase):
    name = "max_pool3d"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 8,
        input_size: Tuple[int, int, int] = (8, 16, 16),
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.input_size = input_size
        self.batch_size = batch_size

        if tag:
            self.name = f"max_pool3d_{tag}"
        else:
            parts = ["max_pool3d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["MaxPool3dTest"]:
        return [
            # Fast path: kernel == stride
            cls(kernel_size=2, stride=2),
            # General path: overlapping windows with padding
            cls(kernel_size=3, stride=2, padding=1),
            # Batch > 1
            cls(kernel_size=2, stride=2, batch_size=2, tag="batch2"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, tag="c1"),
            # Non-cubic kernel/stride
            cls(
                kernel_size=(2, 2, 4),
                stride=(2, 2, 4),
                input_size=(8, 16, 16),
                tag="noncubic_fast",
            ),
            # Stride > kernel
            cls(
                kernel_size=2, stride=3, input_size=(8, 16, 16), tag="stride_gt_kernel"
            ),
            # stride=None (defaults to kernel_size)
            cls(kernel_size=2, stride=None, tag="stride_none"),
            # Global pooling: kernel == spatial
            cls(
                kernel_size=(8, 16, 16),
                stride=(8, 16, 16),
                input_size=(8, 16, 16),
                tag="global",
            ),
            # Non-cubic general path (stride != kernel)
            cls(
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                input_size=(8, 16, 16),
                tag="noncubic_general",
            ),
            # Non-cubic input with cubic kernel
            cls(kernel_size=2, stride=2, input_size=(4, 8, 16), tag="nonsquare_input"),
        ]

    def create_model(self) -> nn.Module:
        return MaxPool3dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(
                self.batch_size,
                self.in_channels,
                *self.input_size,
            ),
        )


class AvgPool1dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class AvgPool1dTest(OpTestCase):
    name = "avg_pool1d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 8,
        seq_len: int = 32,
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.batch_size = batch_size

        if tag:
            self.name = f"avg_pool1d_{tag}"
        else:
            parts = ["avg_pool1d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["AvgPool1dTest"]:
        return [
            # Fast path: kernel == stride
            cls(kernel_size=2, stride=2),
            # General path: overlapping windows with padding
            cls(kernel_size=3, stride=2, padding=1),
            # Fast path: larger kernel
            cls(kernel_size=4, stride=4, seq_len=64),
            # stride=None (defaults to kernel_size)
            cls(kernel_size=4, stride=None, seq_len=64, tag="stride_none"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, tag="c1"),
            # Global pooling
            cls(kernel_size=32, stride=32, tag="global"),
            # Batch > 1
            cls(kernel_size=2, stride=2, batch_size=4, tag="batch4"),
            # Stride > kernel (gaps between windows)
            cls(kernel_size=2, stride=3, seq_len=32, tag="stride_gt_kernel"),
        ]

    def create_model(self) -> nn.Module:
        return AvgPool1dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.batch_size, self.in_channels, self.seq_len),)


class AvgPool2dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class AvgPool2dTest(OpTestCase):
    name = "avg_pool2d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 16,
        input_size: Tuple[int, int] = (32, 32),
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.input_size = input_size
        self.batch_size = batch_size

        if tag:
            self.name = f"avg_pool2d_{tag}"
        else:
            parts = ["avg_pool2d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            parts.append(f"{input_size[0]}x{input_size[1]}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["AvgPool2dTest"]:
        return [
            # Fast path: kernel == stride
            cls(kernel_size=2, stride=2, input_size=(32, 32)),
            # General path: overlapping windows
            cls(kernel_size=3, stride=2, padding=1, input_size=(32, 32)),
            # Fast path: 4x4 pooling
            cls(kernel_size=4, stride=4, input_size=(64, 64)),
            # General path: stride != kernel
            cls(kernel_size=3, stride=1, input_size=(16, 16)),
            # Batch > 1
            cls(kernel_size=2, stride=2, input_size=(32, 32), batch_size=4),
            # stride=None
            cls(kernel_size=2, stride=None, input_size=(32, 32), tag="stride_none"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, input_size=(16, 16), tag="c1"),
            # Global pooling
            cls(kernel_size=8, stride=8, input_size=(8, 8), tag="global"),
            # Non-square kernel/stride
            cls(
                kernel_size=(2, 3),
                stride=(2, 3),
                input_size=(16, 18),
                tag="nonsquare_fast",
            ),
            cls(
                kernel_size=(3, 2),
                stride=(1, 2),
                padding=(1, 0),
                input_size=(16, 16),
                tag="nonsquare_general",
            ),
            # Stride > kernel (gaps between windows)
            cls(kernel_size=2, stride=3, input_size=(16, 16), tag="stride_gt_kernel"),
            # Non-square input with square kernel
            cls(kernel_size=2, stride=2, input_size=(16, 32), tag="nonsquare_input"),
        ]

    def create_model(self) -> nn.Module:
        return AvgPool2dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(
                self.batch_size,
                self.in_channels,
                self.input_size[0],
                self.input_size[1],
            ),
        )


class AvgPool3dModel(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


@register_test
class AvgPool3dTest(OpTestCase):
    name = "avg_pool3d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        padding=0,
        in_channels: int = 8,
        input_size: Tuple[int, int, int] = (8, 16, 16),
        batch_size: int = 1,
        tag: str = "",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.input_size = input_size
        self.batch_size = batch_size

        if tag:
            self.name = f"avg_pool3d_{tag}"
        else:
            parts = ["avg_pool3d", f"k{kernel_size}", f"s{stride}"]
            if padding != 0:
                parts.append(f"p{padding}")
            self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["AvgPool3dTest"]:
        return [
            # Fast path: kernel == stride
            cls(kernel_size=2, stride=2),
            # General path: overlapping windows with padding
            cls(kernel_size=3, stride=2, padding=1),
            # Batch > 1
            cls(kernel_size=2, stride=2, batch_size=2, tag="batch2"),
            # Single channel
            cls(kernel_size=2, stride=2, in_channels=1, tag="c1"),
            # Non-cubic kernel/stride
            cls(
                kernel_size=(2, 2, 4),
                stride=(2, 2, 4),
                input_size=(8, 16, 16),
                tag="noncubic_fast",
            ),
            # Stride > kernel
            cls(
                kernel_size=2, stride=3, input_size=(8, 16, 16), tag="stride_gt_kernel"
            ),
            # stride=None (defaults to kernel_size)
            cls(kernel_size=2, stride=None, tag="stride_none"),
            # Global pooling: kernel == spatial
            cls(
                kernel_size=(8, 16, 16),
                stride=(8, 16, 16),
                input_size=(8, 16, 16),
                tag="global",
            ),
            # Non-cubic general path (stride != kernel)
            cls(
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
                input_size=(8, 16, 16),
                tag="noncubic_general",
            ),
            # Non-cubic input with cubic kernel
            cls(kernel_size=2, stride=2, input_size=(4, 8, 16), tag="nonsquare_input"),
        ]

    def create_model(self) -> nn.Module:
        return AvgPool3dModel(self.kernel_size, self.stride, self.padding)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(
                self.batch_size,
                self.in_channels,
                *self.input_size,
            ),
        )


# =============================================================================
# RMS NORM (torch.nn.functional.rms_norm)
# =============================================================================


class RMSNormModel(nn.Module):
    """Model using torch.nn.functional.rms_norm."""

    def __init__(self, hidden_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x, (self.hidden_dim,), self.weight, self.eps
        )


@register_test
class RMSNormTest(OpTestCase):
    """Test case for torch.nn.functional.rms_norm (aten.rms_norm)."""

    name = "aten_rms_norm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        hidden_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        eps: float = 1e-5,
    ):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eps = eps
        self.name = "aten_rms_norm"

    @classmethod
    def get_test_configs(cls) -> List["RMSNormTest"]:
        return [
            cls(),
            cls(hidden_dim=128, eps=1e-6),
        ]

    def create_model(self) -> nn.Module:
        return RMSNormModel(self.hidden_dim, self.eps)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        return (x,)


# =============================================================================
# ROPE OP (Custom Op)
# =============================================================================


class RopeModel(nn.Module):
    """Model that applies RoPE with dynamic position."""

    def __init__(
        self,
        dims: int = 64,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = pos_tensor.item()
        q_rot = torch.ops.mlx.rope(
            q, self.dims, pos, self.traditional, self.base, self.scale, None
        )
        k_rot = torch.ops.mlx.rope(
            k, self.dims, pos, self.traditional, self.base, self.scale, None
        )
        return q_rot, k_rot


@register_test
class RopeTest(OpTestCase):
    """Test case for RoPE."""

    name = "rope"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 8,
        seq_len: int = 16,
        head_dim: int = 64,
        dims: Optional[int] = None,
        pos: int = 0,
        traditional: bool = False,
        base: float = 500000.0,
        scale: float = 1.0,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dims = dims if dims is not None else head_dim
        self.pos = pos
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.name = "rope"

    @classmethod
    def get_test_configs(cls) -> List["RopeTest"]:
        return [
            cls(),
            cls(traditional=True),
            cls(head_dim=64, dims=32),
            cls(head_dim=64, dims=32, traditional=True),
        ]

    def create_model(self) -> nn.Module:
        return RopeModel(
            dims=self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        pos_tensor = torch.tensor(self.pos, dtype=torch.int64)
        return (q, k, pos_tensor)


# =============================================================================
# SLICE UPDATE OP (Custom Op)
# =============================================================================


class SliceUpdateModel(nn.Module):
    """Slice update using llama.update_cache custom op."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,
        dynamic_pos: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos
        self.dynamic_pos = dynamic_pos

        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dynamic_pos:
            pos = start_pos.item()
        else:
            pos = self.start_pos

        torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        return self.k_cache.clone(), self.v_cache.clone()


@register_test
class SliceUpdateTest(OpTestCase):
    """Test case for slice update operations."""

    name = "slice_update"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        seq_step: int = 8,
        start_pos: int = 0,
        test_start_pos: int = 16,
        export_seq_step: int = 8,
        test_seq_step: int = 4,
        variant: str = "static",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.seq_step = seq_step
        self.start_pos = start_pos
        self.test_start_pos = test_start_pos
        self.export_seq_step = export_seq_step
        self.test_seq_step = test_seq_step
        self.variant = variant

        variant_names = {
            "static": "slice_update",
            "dynamic_pos": "slice_update_dynamic_pos",
            "fully_dynamic": "slice_update_fully_dynamic",
        }
        self.name = variant_names.get(variant, "slice_update")

        if variant == "fully_dynamic":
            self.seq_dim = Dim("seq_step", min=1, max=max_seq_len)
        else:
            self.seq_dim = None

    @classmethod
    def get_test_configs(cls) -> List["SliceUpdateTest"]:
        return [
            cls(variant="static"),
            cls(variant="dynamic_pos"),
            cls(variant="fully_dynamic"),
        ]

    def _has_dynamic_pos(self) -> bool:
        return self.variant in ("dynamic_pos", "fully_dynamic")

    def _has_dynamic_seq(self) -> bool:
        return self.variant == "fully_dynamic"

    def create_model(self) -> nn.Module:
        return SliceUpdateModel(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            start_pos=self.start_pos,
            dynamic_pos=self._has_dynamic_pos(),
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        seq_len = self.export_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(0, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        seq_len = self.test_seq_step if self._has_dynamic_seq() else self.seq_step
        k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

        if self._has_dynamic_pos():
            start_pos = torch.tensor(self.test_start_pos, dtype=torch.int64)
            return (k_val, v_val, start_pos)
        return (k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        if self.variant == "fully_dynamic":
            return {
                "k_val": {1: self.seq_dim},
                "v_val": {1: self.seq_dim},
                "start_pos": None,
            }
        return None


# =============================================================================
# KV CACHE OPS (Custom Op)
# =============================================================================


class KVCacheUpdateModel(nn.Module):
    """KV cache update using llama.update_cache with transpose pattern."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos

        self.register_buffer(
            "k_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, num_heads, max_seq_len, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_pos = self.start_pos

        k_val_transposed = k_val.transpose(1, 2)
        v_val_transposed = v_val.transpose(1, 2)

        k_cache_view = self.k_cache.transpose(1, 2)
        v_cache_view = self.v_cache.transpose(1, 2)

        _ = torch.ops.llama.update_cache(k_val_transposed, k_cache_view, start_pos)
        _ = torch.ops.llama.update_cache(v_val_transposed, v_cache_view, start_pos)

        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelDirect(nn.Module):
    """KV cache update using llama.update_cache directly without transpose."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        start_pos: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos

        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_pos = self.start_pos

        torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)

        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelDynamicPos(nn.Module):
    """KV cache update with dynamic start_pos input."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = start_pos.item()

        torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        return self.k_cache.clone(), self.v_cache.clone()


class KVCacheUpdateModelFullyDynamic(nn.Module):
    """KV cache update with dynamic start_pos and sequence length."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.register_buffer(
            "k_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, max_seq_len, num_heads, head_dim)
        )

    def forward(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = start_pos.item()

        torch.ops.llama.update_cache(k_val, self.k_cache, pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, pos)

        return self.k_cache.clone(), self.v_cache.clone()


@register_test
class KVCacheTest(OpTestCase):
    """Test case for KV cache update operations."""

    name = "kv_cache"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        max_seq_len: int = 128,
        seq_step: int = 8,
        start_pos: int = 0,
        test_start_pos: int = 16,
        export_seq_step: int = 8,
        test_seq_step: int = 4,
        variant: str = "pattern",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.seq_step = seq_step
        self.start_pos = start_pos
        self.test_start_pos = test_start_pos
        self.export_seq_step = export_seq_step
        self.test_seq_step = test_seq_step
        self.variant = variant

        variant_names = {
            "pattern": "kv_cache",
            "direct": "kv_cache_direct",
            "dynamic_pos": "kv_cache_dynamic_pos",
            "fully_dynamic": "kv_cache_fully_dynamic",
        }
        self.name = variant_names.get(variant, "kv_cache")

        # Create dynamic dimension for fully_dynamic variant
        if variant == "fully_dynamic":
            self.seq_dim = Dim("seq_step", min=1, max=max_seq_len)
        else:
            self.seq_dim = None

    @classmethod
    def get_test_configs(cls) -> List["KVCacheTest"]:
        # Note: "pattern" variant is covered by KVCachePatternTest
        return [
            cls(variant="direct"),
            cls(variant="dynamic_pos"),
            cls(variant="fully_dynamic"),
        ]

    def _has_dynamic_pos(self) -> bool:
        return self.variant in ("dynamic_pos", "fully_dynamic")

    def _has_dynamic_seq(self) -> bool:
        return self.variant == "fully_dynamic"

    def create_model(self) -> nn.Module:
        if self.variant == "direct":
            return KVCacheUpdateModelDirect(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                start_pos=self.start_pos,
            )
        elif self.variant == "dynamic_pos":
            return KVCacheUpdateModelDynamicPos(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )
        elif self.variant == "fully_dynamic":
            return KVCacheUpdateModelFullyDynamic(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )
        return KVCacheUpdateModel(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            start_pos=self.start_pos,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.variant == "pattern":
            k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
            v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
            return (k_val, v_val)
        else:
            # direct, dynamic_pos, fully_dynamic use [B, S, H, D] layout
            seq_len = self.export_seq_step if self._has_dynamic_seq() else self.seq_step
            k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
            v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

            if self._has_dynamic_pos():
                start_pos = torch.tensor(0, dtype=torch.int64)
                return (k_val, v_val, start_pos)
            return (k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.variant == "pattern":
            k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
            v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
            return (k_val, v_val)
        else:
            seq_len = self.test_seq_step if self._has_dynamic_seq() else self.seq_step
            k_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)
            v_val = torch.randn(1, seq_len, self.num_heads, self.head_dim)

            if self._has_dynamic_pos():
                start_pos = torch.tensor(self.test_start_pos, dtype=torch.int64)
                return (k_val, v_val, start_pos)
            return (k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict]:
        if self.variant == "fully_dynamic":
            return {
                "k_val": {1: self.seq_dim},
                "v_val": {1: self.seq_dim},
                "start_pos": None,
            }
        return None


# =============================================================================
# ET KV CACHE OPS (Functional cache update)
# =============================================================================

from executorch.backends.apple.mlx.examples.cache import ETKVCache


class ETKVCacheModel(nn.Module):
    """
    Test model wrapping ETKVCache from cache.py.

    This tests the ExecutorTorch llama KVCache-compatible interface that uses
    the mlx::kv_cache_update op internally.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool = True,
    ):
        super().__init__()
        self.cache = ETKVCache(
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=enable_dynamic_shape,
        )

    def forward(
        self,
        input_pos: torch.Tensor,  # [S] position tensor
        k_val: torch.Tensor,  # [B, H, S, D]
        v_val: torch.Tensor,  # [B, H, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache using the llama KVCache interface."""
        return self.cache.update(input_pos, k_val, v_val)


@register_test
class ETKVCacheTest(OpTestCase):
    """
    Test case for ETKVCache with ExecutorTorch llama KVCache interface.

    This verifies that ETKVCache:
    1. Follows the same interface as examples/models/llama/attention.py KVCache
    2. Uses mlx::kv_cache_update internally
    3. Works correctly with dynamic shapes
    """

    name = "et_kv_cache"
    rtol = 1e-5
    atol = 1e-5
    expected_node_counts = {
        "ItemIntNode": 1,
        "SymSizeNode": 2,
        "AddIntNode": 2,
        "SliceUpdateNode": 2,
        "IdCopyNode": 2,
    }

    def __init__(
        self,
        max_batch_size: int = 1,
        n_heads: int = 4,
        head_dim: int = 64,
        max_context_length: int = 128,
        seq_step: int = 8,
        enable_dynamic_shape: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.seq_step = seq_step
        self.enable_dynamic_shape = enable_dynamic_shape

    @classmethod
    def get_test_configs(cls) -> List["ETKVCacheTest"]:
        return [
            cls(),  # default config
            cls(n_heads=8, head_dim=32),  # different head config
            cls(enable_dynamic_shape=False),  # static shape mode
        ]

    def create_model(self) -> nn.Module:
        return ETKVCacheModel(
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_context_length,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=self.enable_dynamic_shape,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Note: ETKVCache.update() takes (input_pos, k_val, v_val) - position first
        input_pos = torch.tensor([0], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Test with different position and different seq_step
        test_seq_step = self.seq_step + 4
        input_pos = torch.tensor([16], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        # seq_step (dim 2) is dynamic for k_val and v_val
        seq_dim = Dim("seq_step", min=1, max=self.max_context_length)
        return {
            "input_pos": None,
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
        }


class ETKVCacheIntModel(nn.Module):
    """
    Test model that passes int/SymInt (not tensor) to ETKVCache.update().

    This tests the "int route" where the caller extracts the start position
    from the tensor before calling update, which is the preferred pattern
    in multi-layer models to avoid redundant SymInt extraction.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool = True,
    ):
        super().__init__()
        self.cache = ETKVCache(
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=enable_dynamic_shape,
        )

    def forward(
        self,
        input_pos: torch.Tensor,  # [S] position tensor
        k_val: torch.Tensor,  # [B, H, S, D]
        v_val: torch.Tensor,  # [B, H, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract int from tensor, then pass to update (the int route)."""
        start_pos = input_pos[0].item()
        return self.cache.update(start_pos, k_val, v_val)


@register_test
class ETKVCacheIntTest(OpTestCase):
    """
    Test case for ETKVCache with int/SymInt input_pos.

    This verifies the "int route" where the caller extracts the start position
    before calling update, matching the recommended pattern for multi-layer models.
    """

    name = "et_kv_cache_int"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        max_batch_size: int = 1,
        n_heads: int = 4,
        head_dim: int = 64,
        max_context_length: int = 128,
        seq_step: int = 8,
        enable_dynamic_shape: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.seq_step = seq_step
        self.enable_dynamic_shape = enable_dynamic_shape

    @classmethod
    def get_test_configs(cls) -> List["ETKVCacheIntTest"]:
        return [
            cls(),  # default config
            cls(n_heads=8, head_dim=32),  # different head config
        ]

    def create_model(self) -> nn.Module:
        return ETKVCacheIntModel(
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_context_length,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=self.enable_dynamic_shape,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        input_pos = torch.tensor([0], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        test_seq_step = self.seq_step + 4
        input_pos = torch.tensor([16], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        seq_dim = Dim("seq_step", min=1, max=self.max_context_length)
        return {
            "input_pos": None,
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
        }


class ETKVCacheSliceModel(nn.Module):
    """
    Test model that updates ETKVCache then slices the result.

    This tests that operations on the returned cache work correctly,
    matching the pattern used in attention implementations.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool = True,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.cache = ETKVCache(
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=enable_dynamic_shape,
        )

    def forward(
        self,
        input_pos: torch.Tensor,  # [S] position tensor
        k_val: torch.Tensor,  # [B, H, S, D]
        v_val: torch.Tensor,  # [B, H, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return sliced result."""
        start_pos = input_pos[0].item()
        seq_len = k_val.size(2)
        end_pos = start_pos + seq_len

        torch._check_is_size(start_pos)
        torch._check(end_pos <= self.max_context_length)
        torch._check(end_pos >= 0)

        k_cache, v_cache = self.cache.update(input_pos, k_val, v_val)

        k_valid = k_cache[:, :, :end_pos, :]
        v_valid = v_cache[:, :, :end_pos, :]
        return k_valid, v_valid


@register_test
class ETKVCacheSliceTest(OpTestCase):
    """
    Test case for ETKVCache update followed by slicing.

    This verifies that:
    1. The ET llama KVCache-compatible interface works correctly
    2. Subsequent slice operations on the returned cache work correctly
    """

    name = "et_kv_cache_slice"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        max_batch_size: int = 1,
        n_heads: int = 4,
        head_dim: int = 64,
        max_context_length: int = 128,
        seq_step: int = 8,
        enable_dynamic_shape: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.seq_step = seq_step
        self.enable_dynamic_shape = enable_dynamic_shape

    @classmethod
    def get_test_configs(cls) -> List["ETKVCacheSliceTest"]:
        return [
            cls(),
            cls(n_heads=8, head_dim=32),
        ]

    def create_model(self) -> nn.Module:
        return ETKVCacheSliceModel(
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_context_length,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=self.enable_dynamic_shape,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        input_pos = torch.tensor([0], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, self.seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        test_seq_step = self.seq_step + 4
        input_pos = torch.tensor([16], dtype=torch.int64)
        k_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        v_val = torch.randn(
            self.max_batch_size, self.n_heads, test_seq_step, self.head_dim
        )
        return (input_pos, k_val, v_val)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        seq_dim = Dim("seq_step", min=1, max=self.max_context_length)
        return {
            "input_pos": None,
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
        }


# =============================================================================
# MLX STATIC CACHE TESTS (HuggingFace-compatible interface)
# =============================================================================


class MockModelConfig:
    """
    Mock HuggingFace model config for testing HFStaticCache.

    This simulates the config structure expected by HFStaticCache.
    """

    def __init__(
        self,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        hidden_size: int = 256,
        head_dim: int | None = None,
        max_position_embeddings: int = 128,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.max_position_embeddings = max_position_embeddings

    def get_text_config(self):
        """Return self for HF StaticCache compatibility."""
        return self


class HFStaticCacheModel(nn.Module):
    """
    Test model wrapping HFStaticCache from cache.py.

    This tests the HuggingFace-compatible StaticCache interface.
    """

    def __init__(
        self,
        config: MockModelConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        from executorch.backends.apple.mlx.examples.cache import HFStaticCache

        self.cache = HFStaticCache(config)
        self.layer_idx = layer_idx

        # Register buffers explicitly so torch.export treats them as mutable
        # buffers rather than constants. This mirrors what replace_hf_cache_with_mlx() does.
        for i, layer_cache in enumerate(self.cache.kv_cache):
            self.register_buffer(
                f"key_cache_{i}", layer_cache.k_cache, persistent=False
            )
            self.register_buffer(
                f"value_cache_{i}", layer_cache.v_cache, persistent=False
            )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, H, S, D]
        v_val: torch.Tensor,  # [B, H, S, D]
        cache_position: torch.Tensor,  # 1D tensor with start position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache using HuggingFace-style interface."""
        return self.cache.update(
            k_val,
            v_val,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )


@register_test
class HFStaticCacheTest(OpTestCase):
    """Test case for HFStaticCache with HuggingFace-compatible interface."""

    name = "hf_static_cache"
    rtol = 1e-5
    atol = 1e-5
    expected_node_counts = {
        "ItemIntNode": 1,
        "SymSizeNode": 2,
        "AddIntNode": 2,
        "SliceUpdateNode": 2,
        "IdCopyNode": 2,
    }

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 128,
        seq_step: int = 8,
        layer_idx: int = 0,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_step = seq_step
        self.layer_idx = layer_idx

    @classmethod
    def get_test_configs(cls) -> List["HFStaticCacheTest"]:
        return [
            cls(),  # default config, layer 0
            cls(num_heads=8, head_dim=32, layer_idx=1),  # different config, layer 1
        ]

    def create_model(self) -> nn.Module:
        config = MockModelConfig(
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            hidden_size=self.num_heads * self.head_dim,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_seq_len,
        )
        return HFStaticCacheModel(config, layer_idx=self.layer_idx)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # BHSD layout [B, H, S, D]
        k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        cache_position = torch.tensor([0], dtype=torch.int64)  # 1D tensor
        return (k_val, v_val, cache_position)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Test with different position and different seq_step
        test_seq_step = self.seq_step + 4  # Different from export seq_step
        k_val = torch.randn(1, self.num_heads, test_seq_step, self.head_dim)
        v_val = torch.randn(1, self.num_heads, test_seq_step, self.head_dim)
        cache_position = torch.tensor([16], dtype=torch.int64)  # 1D tensor
        return (k_val, v_val, cache_position)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        # seq_step (dim 2) is dynamic for k_val and v_val
        seq_dim = Dim("seq_step", min=1, max=self.max_seq_len)
        return {
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
            "cache_position": None,
        }


class HFStaticCacheSliceModel(nn.Module):
    """
    Test model that updates HFStaticCache then slices the result.

    This tests that operations on the returned cache work correctly
    with the HuggingFace-compatible interface.
    """

    def __init__(
        self,
        config: MockModelConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        from executorch.backends.apple.mlx.examples.cache import HFStaticCache

        self.max_seq_len = config.max_position_embeddings
        self.cache = HFStaticCache(config)
        self.layer_idx = layer_idx

        # Register buffers explicitly so torch.export treats them as mutable
        # buffers rather than constants. This mirrors what replace_hf_cache_with_mlx() does.
        for i, layer_cache in enumerate(self.cache.kv_cache):
            self.register_buffer(
                f"key_cache_{i}", layer_cache.k_cache, persistent=False
            )
            self.register_buffer(
                f"value_cache_{i}", layer_cache.v_cache, persistent=False
            )

    def forward(
        self,
        k_val: torch.Tensor,  # [B, H, S, D]
        v_val: torch.Tensor,  # [B, H, S, D]
        cache_position: torch.Tensor,  # 1D tensor with start position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return sliced cache (only the valid portion)."""
        pos = cache_position[0].item()
        seq_len = k_val.size(2)
        end_pos = pos + seq_len

        # Add constraints for dynamic shapes
        torch._check(pos >= 0)
        torch._check(end_pos <= self.max_seq_len)
        torch._check(end_pos >= 0)

        # Update cache using HuggingFace-style interface
        k_cache, v_cache = self.cache.update(
            k_val,
            v_val,
            self.layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )

        # Slice to get only the valid portion [0:end_pos]
        k_valid = k_cache[:, :, :end_pos, :]
        v_valid = v_cache[:, :, :end_pos, :]

        return k_valid, v_valid


@register_test
class HFStaticCacheSliceTest(OpTestCase):
    """
    Test case for HFStaticCache update followed by slicing.

    This verifies that:
    1. The HuggingFace-compatible interface works correctly
    2. Subsequent slice operations on the returned cache work correctly
    """

    name = "hf_static_cache_slice"
    rtol = 1e-5
    atol = 1e-5
    expected_node_counts = {
        "ItemIntNode": 2,
        "SymSizeNode": 3,
        "AddIntNode": 3,
        "SliceUpdateNode": 2,
        "IdCopyNode": 2,
        "SliceNode": 2,
    }

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 128,
        seq_step: int = 8,
        layer_idx: int = 0,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_step = seq_step
        self.layer_idx = layer_idx

    @classmethod
    def get_test_configs(cls) -> List["HFStaticCacheSliceTest"]:
        return [
            cls(),  # default config, layer 0
            cls(num_heads=8, head_dim=32, layer_idx=1),  # different config, layer 1
        ]

    def create_model(self) -> nn.Module:
        config = MockModelConfig(
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            hidden_size=self.num_heads * self.head_dim,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_seq_len,
        )
        return HFStaticCacheSliceModel(config, layer_idx=self.layer_idx)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # BHSD layout [B, H, S, D]
        k_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        v_val = torch.randn(1, self.num_heads, self.seq_step, self.head_dim)
        cache_position = torch.tensor([0], dtype=torch.int64)  # 1D tensor
        return (k_val, v_val, cache_position)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Test with different position and different seq_step
        test_seq_step = self.seq_step + 4  # Different from export seq_step
        k_val = torch.randn(1, self.num_heads, test_seq_step, self.head_dim)
        v_val = torch.randn(1, self.num_heads, test_seq_step, self.head_dim)
        cache_position = torch.tensor([16], dtype=torch.int64)  # 1D tensor
        return (k_val, v_val, cache_position)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        # seq_step (dim 2) is dynamic for k_val and v_val
        seq_dim = Dim("seq_step", min=1, max=self.max_seq_len)
        return {
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
            "cache_position": None,
        }


# =============================================================================
# ADDITIONAL ARANGE OPS (Dynamic)
# =============================================================================


class DynamicArangeModel(nn.Module):
    """Model that uses arange with dynamic start/stop from tensor.item()."""

    def __init__(self, length: int, vocab_size: int = 32):
        super().__init__()
        self.length = length
        self.embed = nn.Embedding(vocab_size, 16)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        torch._check(pos.numel() == 1)
        pos_int = pos.item()
        torch._check_is_size(pos_int)
        positions = torch.arange(
            pos_int, pos_int + self.length, device=pos.device, dtype=torch.long
        )
        return self.embed(positions)


@register_test
class DynamicArangeTest(OpTestCase):
    """Test case for torch.arange() with dynamic start/stop."""

    name = "arange_dynamic"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        position: int = 4,
        length: int = 4,
        vocab_size: int = 32,
    ):
        self.position = position
        self.length = length
        self.vocab_size = vocab_size
        self.name = f"arange_dynamic_pos{position}_len{length}"

    @classmethod
    def get_test_configs(cls) -> List["DynamicArangeTest"]:
        return [
            cls(position=0, length=4),
            cls(position=4, length=4),
            cls(position=10, length=8),
        ]

    def create_model(self) -> nn.Module:
        return DynamicArangeModel(length=self.length, vocab_size=self.vocab_size)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        pos = torch.tensor([self.position], dtype=torch.long)
        return (pos,)


class LayerNormModel(nn.Module):
    """Simple model using LayerNorm."""

    def __init__(self, normalized_shape: int = 64, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


@register_test
class LayerNormTest(OpTestCase):
    """Test case for nn.LayerNorm."""

    name = "layer_norm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        normalized_shape: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        eps: float = 1e-5,
    ):
        self.normalized_shape = normalized_shape
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eps = eps
        self.name = "layer_norm"

    @classmethod
    def get_test_configs(cls) -> List["LayerNormTest"]:
        return [
            cls(),
            cls(normalized_shape=128, eps=1e-6),
        ]

    def create_model(self) -> nn.Module:
        return LayerNormModel(self.normalized_shape, self.eps)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.seq_len, self.normalized_shape)
        return (x,)


class Conv1dModel(nn.Module):
    """Simple model using Conv1d."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class Conv1dTest(OpTestCase):
    """Test case for nn.Conv1d."""

    name = "conv1d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        batch_size: int = 2,
        seq_len: int = 64,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch_size = batch_size
        self.seq_len = seq_len

        parts = ["conv1d"]
        if not bias:
            parts.append("no_bias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["Conv1dTest"]:
        return [
            cls(),
            cls(bias=False),
        ]

    def create_model(self) -> nn.Module:
        return Conv1dModel(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.in_channels, self.seq_len)
        return (x,)


class Conv2DModel(nn.Module):
    """Model that performs 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class Conv2DTest(OpTestCase):
    """Test case for conv2d op."""

    name = "conv2d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        input_size: Tuple[int, int] = (32, 32),
        batch_size: int = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.bias = bias

        parts = [
            "conv2d",
            f"in{in_channels}",
            f"out{out_channels}",
            f"k{kernel_size}",
        ]
        if stride != 1:
            parts.append(f"s{stride}")
        if padding != 0:
            parts.append(f"p{padding}")
        parts.append(f"{input_size[0]}x{input_size[1]}")
        if batch_size != 1:
            parts.append(f"b{batch_size}")
        if not bias:
            parts.append("nobias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["Conv2DTest"]:
        return [
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
            ),
            cls(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                input_size=(64, 64),
            ),
            cls(in_channels=64, out_channels=128, kernel_size=1, input_size=(16, 16)),
            # 5x5 conv
            cls(
                in_channels=3,
                out_channels=8,
                kernel_size=5,
                padding=2,
                input_size=(28, 28),
            ),
            # Batch size > 1
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
                batch_size=4,
            ),
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
                bias=False,
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.in_channels, self.input_size[0], self.input_size[1]
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return Conv2DModel(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        )


class Conv3DModel(nn.Module):
    """Model that performs 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class Conv3DTest(OpTestCase):
    """Test case for conv3d op."""

    name = "conv3d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        input_size: Tuple[int, int, int] = (8, 16, 16),
        batch_size: int = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.bias = bias

        parts = [
            "conv3d",
            f"in{in_channels}",
            f"out{out_channels}",
            f"k{kernel_size}",
        ]
        if stride != 1:
            parts.append(f"s{stride}")
        if padding != 0:
            parts.append(f"p{padding}")
        parts.append(f"{input_size[0]}x{input_size[1]}x{input_size[2]}")
        if batch_size != 1:
            parts.append(f"b{batch_size}")
        if not bias:
            parts.append("nobias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["Conv3DTest"]:
        return [
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(8, 16, 16),
            ),
            cls(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                input_size=(8, 16, 16),
            ),
            cls(in_channels=64, out_channels=128, kernel_size=1, input_size=(4, 8, 8)),
            # Batch size > 1
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(8, 16, 16),
                batch_size=2,
            ),
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(8, 16, 16),
                bias=False,
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.input_size[0],
            self.input_size[1],
            self.input_size[2],
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return Conv3DModel(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        )


# =============================================================================
# MATRIX MULTIPLICATION OPS
# =============================================================================


class BmmModel(nn.Module):
    """Model that performs batch matrix multiplication."""

    def __init__(self, batch_size: int, n: int, m: int, p: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(batch_size, m, p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight)


@register_test
class BmmTest(OpTestCase):
    """Test case for bmm (batch matrix multiplication)."""

    name = "bmm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 4,
        n: int = 8,
        m: int = 16,
        p: int = 32,
    ):
        self.batch_size = batch_size
        self.n = n
        self.m = m
        self.p = p
        self.name = f"bmm_{batch_size}x{n}x{m}x{p}"

    @classmethod
    def get_test_configs(cls) -> List["BmmTest"]:
        return [
            cls(batch_size=4, n=8, m=16, p=32),
            cls(batch_size=2, n=64, m=64, p=32),
        ]

    def create_model(self) -> nn.Module:
        return BmmModel(self.batch_size, self.n, self.m, self.p)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.n, self.m)
        return (x,)


class AddmmModel(nn.Module):
    """Model that performs addmm: bias + (mat1 @ mat2)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(
                self.bias, x, self.weight.t(), beta=self.beta, alpha=self.alpha
            )
        else:
            return torch.mm(x, self.weight.t())


@register_test
class AddmmTest(OpTestCase):
    """Test case for addmm."""

    name = "addmm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 2,
        in_features: int = 64,
        out_features: int = 32,
        bias: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        # Build unique test name
        if not bias:
            name = f"addmm_{in_features}x{out_features}_no_bias"
        elif alpha != 1.0 or beta != 1.0:
            name = f"addmm_{in_features}x{out_features}_a{alpha}_b{beta}"
        else:
            name = f"addmm_{in_features}x{out_features}"
        self.name = name

    @classmethod
    def get_test_configs(cls) -> List["AddmmTest"]:
        return [
            cls(
                batch_size=2, in_features=64, out_features=32
            ),  # with bias, default alpha/beta
            cls(
                batch_size=2, in_features=64, out_features=32, bias=False
            ),  # without bias
            cls(batch_size=4, in_features=128, out_features=64),  # larger size
            cls(
                batch_size=2, in_features=64, out_features=32, alpha=2.0, beta=0.5
            ),  # custom alpha/beta
        ]

    def create_model(self) -> nn.Module:
        return AddmmModel(
            self.in_features,
            self.out_features,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.in_features)
        return (x,)


# =============================================================================
# EXPAND AND INDEX OPS
# =============================================================================


class ExpandModel(nn.Module):
    """Model that expands a tensor to a larger shape."""

    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.target_shape)


@register_test
class ExpandTest(OpTestCase):
    """Test case for expand (expand_copy) op."""

    name = "expand"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 1),
        target_shape: Tuple[int, ...] = (2, 3, 4),
    ):
        self.input_shape = input_shape
        self.target_shape = target_shape

        input_str = "x".join(str(s) for s in input_shape)
        target_str = "x".join(str(s) for s in target_shape)
        self.name = f"expand_{input_str}_to_{target_str}"

    @classmethod
    def get_test_configs(cls) -> List["ExpandTest"]:
        return [
            cls(input_shape=(2, 3, 1), target_shape=(2, 3, 4)),
            cls(input_shape=(1, 3, 4), target_shape=(2, 3, 4)),
            cls(input_shape=(1, 1, 4), target_shape=(2, 3, 4)),
            cls(input_shape=(1, 1, 1), target_shape=(2, 3, 4)),
            cls(input_shape=(1, 8), target_shape=(4, 8)),
            cls(input_shape=(1, 1, 1, 64), target_shape=(2, 8, 16, 64)),
            # Expand with -1 (keep dimension unchanged from input)
            cls(input_shape=(93,), target_shape=(1, -1)),
            # Multiple -1 dimensions (keep all but first)
            cls(input_shape=(1, 1, 5, 8), target_shape=(1, -1, -1, -1)),
            # Multiple -1 with actual expansion on first dim
            cls(input_shape=(1, 3, 5, 8), target_shape=(2, -1, -1, -1)),
            # Two -1 dimensions at start
            cls(input_shape=(2, 3, 4), target_shape=(-1, -1, 4)),
        ]

    def create_model(self) -> nn.Module:
        return ExpandModel(self.target_shape)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape)
        return (x,)


class IndexModel(nn.Module):
    """Model that indexes a tensor using another tensor."""

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x[indices]


@register_test
class IndexTest(OpTestCase):
    """Test case for tensor indexing."""

    name = "index"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        table_size: int = 100,
        num_indices: int = 10,
    ):
        self.table_size = table_size
        self.num_indices = num_indices
        self.name = f"index_{table_size}_idx{num_indices}"

    @classmethod
    def get_test_configs(cls) -> List["IndexTest"]:
        return [
            cls(table_size=100, num_indices=10),
            cls(table_size=50, num_indices=5),
        ]

    def create_model(self) -> nn.Module:
        return IndexModel()

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.table_size)
        indices = torch.randint(0, self.table_size, (self.num_indices,))
        return (x, indices)


class IndexUpdateModel(nn.Module):
    """Model that performs index_copy on a mutable buffer.

    This triggers the INDEX_UPDATE pattern which matches aten.index_copy.default
    on a mutable buffer and lowers it to IndexUpdateNode.
    """

    def __init__(
        self,
        buffer_size: int = 128,
        feature_dim: int = 64,
        axis: int = 0,
    ):
        super().__init__()
        self.axis = axis
        if axis == 0:
            self.register_buffer("data", torch.zeros(buffer_size, feature_dim))
        else:
            # axis == 1
            self.register_buffer("data", torch.zeros(feature_dim, buffer_size))

    def forward(self, indices: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Update buffer at indices along axis using index_copy."""
        self.data.index_copy_(self.axis, indices, update)
        return self.data.clone()


@register_test
class IndexUpdateTest(OpTestCase):
    """Test case for index_update pattern (index_copy on mutable buffer).

    This tests the INDEX_UPDATE pattern handler which recognizes
    aten.index_copy.default on a mutable buffer and lowers it to IndexUpdateNode.
    The buffer is managed internally by the MLX backend.
    """

    name = "index_update"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self,
        buffer_size: int = 128,
        feature_dim: int = 64,
        num_indices: int = 8,
        axis: int = 0,
    ):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.num_indices = num_indices
        self.axis = axis
        self.name = (
            f"index_update_axis{axis}_{buffer_size}x{feature_dim}_idx{num_indices}"
        )

    @classmethod
    def get_test_configs(cls) -> List["IndexUpdateTest"]:
        return [
            # Basic case: update along axis 0
            cls(buffer_size=128, feature_dim=64, num_indices=8, axis=0),
            # Smaller buffer
            cls(buffer_size=32, feature_dim=16, num_indices=4, axis=0),
            # Update along axis 1
            cls(buffer_size=64, feature_dim=32, num_indices=8, axis=1),
        ]

    def create_model(self) -> nn.Module:
        return IndexUpdateModel(
            buffer_size=self.buffer_size,
            feature_dim=self.feature_dim,
            axis=self.axis,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Create unique indices (no duplicates) for index_copy
        # PyTorch requires int64 (long) for indices
        indices = torch.randperm(self.buffer_size)[: self.num_indices].to(torch.int64)

        # Create update tensor with shape matching the indexed dimension
        if self.axis == 0:
            update = torch.randn(self.num_indices, self.feature_dim)
        else:
            update = torch.randn(self.feature_dim, self.num_indices)

        return (indices, update)


# =============================================================================
# SPLIT OPS
# =============================================================================


class SplitWithSizesModel(nn.Module):
    """Model that splits a tensor into chunks with specified sizes."""

    def __init__(self, sizes, dim=0):
        super().__init__()
        self.sizes = sizes
        self.dim = dim

    def forward(self, x):
        chunks = torch.ops.aten.split_with_sizes_copy.default(x, self.sizes, self.dim)
        return chunks[0]


class SplitWithSizesMultiOutputModel(nn.Module):
    """Model that splits with specified sizes and uses multiple outputs."""

    def __init__(self, sizes, dim=0):
        super().__init__()
        self.sizes = sizes
        self.dim = dim

    def forward(self, x):
        chunks = torch.ops.aten.split_with_sizes_copy.default(x, self.sizes, self.dim)
        return chunks[0] + chunks[-1]


class SplitUniformModel(nn.Module):
    """Model that splits a tensor into chunks of uniform size using torch.split."""

    def __init__(self, split_size, dim=0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_size, dim=self.dim)
        return chunks[0]


class SplitUniformMultiOutputModel(nn.Module):
    """Model that splits uniformly and uses multiple outputs."""

    def __init__(self, split_size, dim=0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_size, dim=self.dim)
        return torch.cat([chunks[0], chunks[-1]], dim=self.dim)


@register_test
class SplitTest(OpTestCase):
    name = "split"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape, model_cls, model_kwargs, tag=""):
        self.shape = shape
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.name = f"split_{tag}" if tag else "split"

    @classmethod
    def get_test_configs(cls) -> List["SplitTest"]:
        return [
            # split_with_sizes_copy tests
            cls(
                shape=(9, 4),
                model_cls=SplitWithSizesModel,
                model_kwargs={"sizes": [2, 3, 4], "dim": 0},
                tag="sizes_dim0",
            ),
            cls(
                shape=(3, 10),
                model_cls=SplitWithSizesModel,
                model_kwargs={"sizes": [2, 3, 5], "dim": 1},
                tag="sizes_dim1",
            ),
            cls(
                shape=(2, 12, 4),
                model_cls=SplitWithSizesModel,
                model_kwargs={"sizes": [3, 4, 5], "dim": 1},
                tag="sizes_3d",
            ),
            cls(
                shape=(8, 4),
                model_cls=SplitWithSizesModel,
                model_kwargs={"sizes": [3, 5], "dim": 0},
                tag="sizes_two",
            ),
            cls(
                shape=(10, 3),
                model_cls=SplitWithSizesMultiOutputModel,
                model_kwargs={"sizes": [5, 5], "dim": 0},
                tag="sizes_multi",
            ),
            # torch.split (uniform) tests
            cls(
                shape=(10, 4),
                model_cls=SplitUniformModel,
                model_kwargs={"split_size": 3, "dim": 0},
                tag="uniform_dim0",
            ),
            cls(
                shape=(3, 7),
                model_cls=SplitUniformModel,
                model_kwargs={"split_size": 4, "dim": 1},
                tag="uniform_dim1",
            ),
            cls(
                shape=(11, 5),
                model_cls=SplitUniformMultiOutputModel,
                model_kwargs={"split_size": 3, "dim": 0},
                tag="uniform_multi",
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape),)

    def create_model(self) -> nn.Module:
        return self.model_cls(**self.model_kwargs)


# =============================================================================
# ARANGE OPS
# =============================================================================


class ArangeModel(nn.Module):
    """Model that creates a tensor using arange and multiplies with input."""

    def __init__(self, stop: int, use_dtype: bool = True):
        super().__init__()
        self.stop = stop
        self.use_dtype = use_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dtype:
            indices = torch.arange(self.stop, dtype=x.dtype, device=x.device)
        else:
            # No dtype - let MLX infer (defaults to int64 for integer inputs)
            indices = torch.arange(self.stop, device=x.device)
            indices = indices.to(x.dtype)  # Cast for multiplication
        return x * indices


@register_test
class ArangeTest(OpTestCase):
    """Test case for torch.arange()."""

    name = "arange"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        stop: int = 10,
        dtype: torch.dtype = torch.float32,
        use_dtype: bool = True,
    ):
        self.stop = stop
        self.dtype = dtype
        self.use_dtype = use_dtype
        dtype_name = str(dtype).split(".")[-1]
        if use_dtype:
            self.name = f"arange_{stop}_{dtype_name}"
        else:
            self.name = f"arange_{stop}_no_dtype"

    @classmethod
    def get_test_configs(cls) -> List["ArangeTest"]:
        return [
            # With explicit dtype
            cls(stop=10, dtype=torch.float32, use_dtype=True),
            cls(stop=32, dtype=torch.float32, use_dtype=True),
            cls(stop=100, dtype=torch.float32, use_dtype=True),
            cls(stop=16, dtype=torch.int32, use_dtype=True),
            cls(stop=16, dtype=torch.int64, use_dtype=True),
            # Without dtype (let MLX infer)
            cls(stop=10, dtype=torch.float32, use_dtype=False),
            cls(stop=32, dtype=torch.float32, use_dtype=False),
        ]

    def create_model(self) -> nn.Module:
        return ArangeModel(self.stop, use_dtype=self.use_dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.dtype in (torch.int32, torch.int64):
            x = torch.randint(1, 10, (self.stop,), dtype=self.dtype)
        else:
            x = torch.randn(self.stop, dtype=self.dtype)
        return (x,)


# =============================================================================
# UNARY MATH OPS
# =============================================================================


class UnaryOpModel(nn.Module):
    """Generic model that applies a single unary torch op."""

    def __init__(self, op_fn: Callable):
        super().__init__()
        self.op_fn = op_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op_fn(x)


def _input_fn(
    uniform: bool = False, scale: float = 1.0, offset: float = 0.0, abs: bool = False
):
    """Return a callable(shape, dtype) that generates a single-element input tuple.

    Args:
        uniform: Use torch.rand (uniform [0,1]) instead of torch.randn (normal).
        scale: Multiply the base tensor by this value.
        offset: Add this value after scaling.
        abs: Apply .abs() to the base tensor before scale/offset.
    """

    def fn(shape, dtype):
        base = (
            torch.rand(shape, dtype=dtype)
            if uniform
            else torch.randn(shape, dtype=dtype)
        )
        if abs:
            base = base.abs()
        return (base * scale + offset,)

    return fn


def _bool_input_fn():
    """Return a callable(shape, dtype) that generates a single-element bool tensor tuple."""

    def fn(shape, _dtype):
        return (torch.randint(0, 2, shape, dtype=torch.bool),)

    return fn


def _int_input_fn(low: int = -100, high: int = 100):
    """Return a callable(shape, dtype) that generates a single-element integer tensor tuple."""

    def fn(shape, dtype):
        return (torch.randint(low, high, shape, dtype=dtype),)

    return fn


# Standard shape and dtype configs used by unary tests.
_SHAPES_3 = [(16,), (4, 4), (2, 3, 4)]
_SHAPES_2 = [(16,), (4, 4)]
_UNARY_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_unary_op_test(
    op_name: str,
    op_fn: Callable,
    shapes: List[Tuple[int, ...]] = None,
    dtypes: List[torch.dtype] = None,
    input_fn: Callable = None,
) -> type:
    """Generate a registered OpTestCase subclass for a unary math op.

    Args:
        op_name: Name used for test registration and output directories.
        op_fn: The torch function to test (e.g. torch.floor).
        shapes: List of input shapes. Defaults to _SHAPES_2.
        dtypes: List of dtypes to test. Defaults to _UNARY_DTYPES.
        input_fn: Callable(shape, dtype) -> Tuple[Tensor, ...] that creates inputs.
                  Defaults to _input_fn() (standard randn).
    """
    if shapes is None:
        shapes = _SHAPES_2
    if dtypes is None:
        dtypes = _UNARY_DTYPES
    if input_fn is None:
        input_fn = _input_fn()

    class _Test(OpTestCase):
        name = op_name

        def __init__(
            self,
            shape: Tuple[int, ...],
            dtype: torch.dtype,
        ):
            self.shape = shape
            self.dtype = dtype
            shape_str = "x".join(str(s) for s in shape)
            dtype_str = str(dtype).replace("torch.", "")
            self.name = f"{op_name}_{shape_str}_{dtype_str}"

        @classmethod
        def get_test_configs(cls) -> List["_Test"]:
            return [cls(shape=s, dtype=d) for s in shapes for d in dtypes]

        def create_inputs(self) -> Tuple[torch.Tensor, ...]:
            return input_fn(self.shape, self.dtype)

        def create_model(self) -> nn.Module:
            return UnaryOpModel(op_fn)

    _Test.__name__ = f"{op_name.title().replace('_', '')}Test"
    _Test.__qualname__ = _Test.__name__
    return _Test


# fmt: off
# Each entry is a dict with required keys "op_name" and "op_fn".
# Optional keys: "shapes" (default _SHAPES_2), "dtypes" (default _UNARY_DTYPES),
#                "input_fn" (default _input_fn()).
# _input_fn(uniform, scale, offset) — uniform=True uses rand, False uses randn.
_UNARY_OP_TESTS = [
    {"op_name": "floor",      "op_fn": torch.floor,      "shapes": _SHAPES_3, "input_fn": _input_fn(scale=10)},
    {"op_name": "ceil",       "op_fn": torch.ceil,       "shapes": _SHAPES_3, "input_fn": _input_fn(scale=10)},
    {"op_name": "square",     "op_fn": torch.square,     "shapes": _SHAPES_3},
    {"op_name": "exp",        "op_fn": torch.exp,        "shapes": _SHAPES_3},
    {"op_name": "sin",        "op_fn": torch.sin,        "shapes": _SHAPES_3, "input_fn": _input_fn(scale=3.14159)},
    {"op_name": "cos",        "op_fn": torch.cos,        "shapes": _SHAPES_3, "input_fn": _input_fn(scale=3.14159)},
    {"op_name": "tan",        "op_fn": torch.tan,        "input_fn": _input_fn(scale=0.5)},
    {"op_name": "asin",       "op_fn": torch.asin,       "input_fn": _input_fn(uniform=True, scale=2, offset=-1)},
    {"op_name": "acos",       "op_fn": torch.acos,       "input_fn": _input_fn(uniform=True, scale=2, offset=-1)},
    {"op_name": "atan",       "op_fn": torch.atan},
    {"op_name": "sinh",       "op_fn": torch.sinh},
    {"op_name": "cosh",       "op_fn": torch.cosh},
    {"op_name": "asinh",      "op_fn": torch.asinh},
    {"op_name": "acosh",      "op_fn": torch.acosh,      "input_fn": _input_fn(uniform=True, offset=1.0)},
    {"op_name": "atanh",      "op_fn": torch.atanh,      "input_fn": _input_fn(uniform=True, scale=1.8, offset=-0.9)},
    {"op_name": "log2",       "op_fn": torch.log2,       "input_fn": _input_fn(uniform=True, offset=0.1)},
    {"op_name": "log10",      "op_fn": torch.log10,      "input_fn": _input_fn(uniform=True, offset=0.1)},
    {"op_name": "log1p",      "op_fn": torch.log1p,      "input_fn": _input_fn(uniform=True)},
    {"op_name": "erf",        "op_fn": torch.erf},
    {"op_name": "expm1",      "op_fn": torch.expm1},
    {"op_name": "round",      "op_fn": torch.round,      "input_fn": _input_fn(scale=10)},
    {"op_name": "reciprocal", "op_fn": torch.reciprocal, "input_fn": _input_fn(offset=1.0)},
    {"op_name": "sqrt",       "op_fn": torch.sqrt,       "input_fn": _input_fn(uniform=True, offset=0.1)},
    {"op_name": "abs",        "op_fn": torch.abs},
    {"op_name": "neg",        "op_fn": torch.neg},
    {"op_name": "logical_not","op_fn": torch.logical_not, "shapes": [(2, 3, 4), (10,), (4, 8)], "dtypes": [torch.bool], "input_fn": _bool_input_fn()},
    # activations
    {"op_name": "relu",    "op_fn": torch.relu,    "shapes": [(2, 3, 4), (10,), (4, 8), (2, 8, 16), (1, 128, 64)], "dtypes": [torch.float32], "input_fn": _input_fn(scale=2, offset=-1)},
    {"op_name": "sigmoid", "op_fn": torch.sigmoid, "shapes": [(2, 3, 4), (10,), (4, 8), (2, 8, 16), (1, 1, 128)],  "dtypes": [torch.float32], "input_fn": _input_fn(scale=2)},
    {"op_name": "tanh",    "op_fn": torch.tanh,    "shapes": [(2, 3, 4), (10,), (4, 8), (2, 8, 16), (1, 1, 128)],  "dtypes": [torch.float32], "input_fn": _input_fn(scale=3)},
    {"op_name": "silu",    "op_fn": nn.SiLU(),     "shapes": [(2, 16, 64), (4, 32, 128)], "dtypes": [torch.float32]},
    # math
    {"op_name": "rsqrt",   "op_fn": torch.rsqrt,   "shapes": [(2, 3, 4), (10,), (4, 8), (2, 8, 16), (1, 64)],     "dtypes": [torch.float32], "input_fn": _input_fn(uniform=True, offset=0.1)},
    {"op_name": "clone",   "op_fn": torch.clone,   "shapes": [(2, 3, 4), (8, 8), (16,)], "dtypes": [torch.float32]},
]
# fmt: on

# Generate and register all unary math op test classes.
for _entry in _UNARY_OP_TESTS:
    _cls = _make_unary_op_test(**_entry)
    register_test(_cls)
    globals()[_cls.__name__] = _cls


# =============================================================================
# BINARY MATH OPS
# =============================================================================


class BinaryOpModel(nn.Module):
    def __init__(self, op_fn: Callable):
        super().__init__()
        self.op_fn = op_fn

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.op_fn(a, b)


class PowerScalarModel(nn.Module):
    def __init__(self, exponent: float):
        super().__init__()
        self.exponent = exponent

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.pow(a, self.exponent)


_BINARY_DTYPES = [torch.float32]


def _make_binary_op_test(
    op_name: str,
    op_fn: Callable,
    shapes: List[Tuple[int, ...]] = None,
    dtypes: List[torch.dtype] = None,
    input_fn_a: Callable = None,
    input_fn_b: Callable = None,
) -> type:
    """Generate a registered OpTestCase subclass for a binary math op."""
    if shapes is None:
        shapes = _SHAPES_3
    if dtypes is None:
        dtypes = _BINARY_DTYPES
    if input_fn_a is None:
        input_fn_a = _input_fn()
    if input_fn_b is None:
        input_fn_b = _input_fn()

    class _Test(OpTestCase):
        name = op_name

        def __init__(
            self,
            shape: Tuple[int, ...],
            dtype: torch.dtype,
        ):
            self.shape = shape
            self.dtype = dtype
            shape_str = "x".join(str(s) for s in shape)
            dtype_str = str(dtype).replace("torch.", "")
            self.name = f"{op_name}_{shape_str}_{dtype_str}"

        @classmethod
        def get_test_configs(cls) -> List["_Test"]:
            return [cls(shape=s, dtype=d) for s in shapes for d in dtypes]

        def create_inputs(self) -> Tuple[torch.Tensor, ...]:
            return input_fn_a(self.shape, self.dtype) + input_fn_b(
                self.shape, self.dtype
            )

        def create_model(self) -> nn.Module:
            return BinaryOpModel(op_fn)

    _Test.__name__ = f"{op_name.title().replace('_', '')}Test"
    _Test.__qualname__ = _Test.__name__
    return _Test


# fmt: off
_BINARY_OP_TESTS = [
    # math
    {"op_name": "maximum",       "op_fn": torch.maximum},
    {"op_name": "minimum",       "op_fn": torch.minimum},
    {"op_name": "atan2",         "op_fn": torch.atan2},
    {"op_name": "logaddexp",     "op_fn": torch.logaddexp},
    {"op_name": "floor_divide",      "op_fn": torch.floor_divide, "input_fn_a": _input_fn(scale=10), "input_fn_b": _input_fn(abs=True, offset=1)},
    {"op_name": "floor_divide_int",  "op_fn": torch.floor_divide, "dtypes": [torch.int32], "input_fn_a": _int_input_fn(-100, 100), "input_fn_b": _int_input_fn(1, 10)},
    {"op_name": "power",         "op_fn": torch.pow,          "input_fn_a": _input_fn(uniform=True, offset=0.5), "input_fn_b": _input_fn(uniform=True, scale=2)},
    # comparison
    {"op_name": "less",          "op_fn": torch.lt, "shapes": [(2, 3, 4), (10,), (4, 8)], "dtypes": [torch.float32, torch.bfloat16]},
    {"op_name": "less_equal",    "op_fn": torch.le, "shapes": [(2, 3, 4), (10,)], "dtypes": [torch.float32]},
    {"op_name": "greater",       "op_fn": torch.gt, "shapes": [(2, 3, 4), (10,)], "dtypes": [torch.float32]},
    {"op_name": "greater_equal", "op_fn": torch.ge, "shapes": [(2, 3, 4), (10,)], "dtypes": [torch.float32]},
    {"op_name": "equal",         "op_fn": torch.eq, "shapes": [(2, 3, 4), (10,)], "dtypes": [torch.float32]},
    {"op_name": "not_equal",     "op_fn": torch.ne, "shapes": [(2, 3, 4), (10,)], "dtypes": [torch.float32]},
    # logical
    {"op_name": "logical_and",   "op_fn": torch.logical_and, "shapes": [(2, 3, 4), (10,), (4, 8)], "dtypes": [torch.bool], "input_fn_a": _bool_input_fn(), "input_fn_b": _bool_input_fn()},
    {"op_name": "logical_or",    "op_fn": torch.logical_or,  "shapes": [(2, 3, 4), (10,), (4, 8)], "dtypes": [torch.bool], "input_fn_a": _bool_input_fn(), "input_fn_b": _bool_input_fn()},
]
# fmt: on


for _entry in _BINARY_OP_TESTS:
    _cls = _make_binary_op_test(**_entry)
    register_test(_cls)
    globals()[_cls.__name__] = _cls


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
            cls(shape=(4, 4), exponent=0.5, dtype=torch.float32),
            cls(shape=(4, 4), exponent=3.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), exponent=-1.0, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.rand(self.shape, dtype=self.dtype) + 0.5,)

    def create_model(self) -> nn.Module:
        return PowerScalarModel(self.exponent)


class CompareScalarModel(nn.Module):
    def __init__(self, op_fn: Callable, scalar: float):
        super().__init__()
        self.op_fn = op_fn
        self.scalar = scalar

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.op_fn(a, self.scalar)


def _make_compare_scalar_test(
    op_name: str,
    op_fn: Callable,
) -> type:
    """Generate a registered OpTestCase subclass for a comparison Scalar op."""

    class _Test(OpTestCase):
        name = op_name

        def __init__(
            self,
            shape: Tuple[int, ...],
            scalar: float,
            dtype: torch.dtype,
        ):
            self.shape = shape
            self.scalar = scalar
            self.dtype = dtype
            shape_str = "x".join(str(s) for s in shape)
            dtype_str = str(dtype).replace("torch.", "")
            self.name = f"{op_name}_{shape_str}_s{scalar}_{dtype_str}"

        @classmethod
        def get_test_configs(cls) -> List["_Test"]:
            return [
                cls(shape=(16,), scalar=0.0, dtype=torch.float32),
                cls(shape=(4, 4), scalar=0.5, dtype=torch.float32),
                cls(shape=(2, 3, 4), scalar=-1.0, dtype=torch.float32),
            ]

        def create_inputs(self) -> Tuple[torch.Tensor, ...]:
            return (torch.randn(self.shape, dtype=self.dtype),)

        def create_model(self) -> nn.Module:
            return CompareScalarModel(op_fn, self.scalar)

    _Test.__name__ = f"{op_name.title().replace('_', '')}Test"
    _Test.__qualname__ = _Test.__name__
    return _Test


_COMPARE_SCALAR_TESTS = [
    {"op_name": "less_scalar", "op_fn": torch.lt},
    {"op_name": "less_equal_scalar", "op_fn": torch.le},
    {"op_name": "greater_scalar", "op_fn": torch.gt},
    {"op_name": "greater_equal_scalar", "op_fn": torch.ge},
    {"op_name": "equal_scalar", "op_fn": torch.eq},
    {"op_name": "not_equal_scalar", "op_fn": torch.ne},
]

for _entry in _COMPARE_SCALAR_TESTS:
    _cls = _make_compare_scalar_test(**_entry)
    register_test(_cls)
    globals()[_cls.__name__] = _cls


# =============================================================================
# REDUCTION OPS
# =============================================================================


class ReductionOpModel(nn.Module):
    def __init__(self, op_fn: Callable, dim=None, keepdim: bool = False):
        super().__init__()
        self.op_fn = op_fn
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return self.op_fn(x)
        return self.op_fn(x, dim=self.dim, keepdim=self.keepdim)


class CorrectionReductionOpModel(nn.Module):
    def __init__(
        self, op_fn: Callable, dim=None, keepdim: bool = False, correction: int = 1
    ):
        super().__init__()
        self.op_fn = op_fn
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            return self.op_fn(x, correction=self.correction)
        return self.op_fn(
            x, dim=self.dim, keepdim=self.keepdim, correction=self.correction
        )


def _make_reduction_op_test(
    op_name: str,
    op_fn: Callable,
    configs: List[dict],
    input_fn: Callable = None,
    has_correction: bool = False,
) -> type:
    """Generate a registered OpTestCase subclass for a reduction op.

    Args:
        op_name: Name used for test registration.
        op_fn: The torch function (e.g. torch.sum).
        configs: List of dicts with keys: shape, dim, keepdim, dtype, and
                 optionally correction (for var/std).
        input_fn: Callable(shape, dtype) -> Tuple[Tensor, ...].
        has_correction: If True, use CorrectionReductionOpModel.
    """
    if input_fn is None:
        input_fn = _input_fn()

    class _Test(OpTestCase):
        name = op_name

        def __init__(self, shape, dim, keepdim, dtype, correction=1):
            self.shape = shape
            self.dim = dim
            self.keepdim = keepdim
            self.dtype = dtype
            self.correction = correction
            shape_str = "x".join(str(s) for s in shape)
            dtype_str = str(dtype).replace("torch.", "")
            dim_str = f"_dim{dim}" if dim is not None else "_all"
            kd_str = "_kd" if keepdim else ""
            corr_str = f"_corr{correction}" if has_correction else ""
            self.name = f"{op_name}_{shape_str}{dim_str}{kd_str}{corr_str}_{dtype_str}"

        @classmethod
        def get_test_configs(cls) -> List["_Test"]:
            return [cls(**c) for c in configs]

        def create_inputs(self) -> Tuple[torch.Tensor, ...]:
            return input_fn(self.shape, self.dtype)

        def create_model(self) -> nn.Module:
            if has_correction:
                return CorrectionReductionOpModel(
                    op_fn,
                    dim=self.dim,
                    keepdim=self.keepdim,
                    correction=self.correction,
                )
            return ReductionOpModel(op_fn, dim=self.dim, keepdim=self.keepdim)

    _Test.__name__ = f"{op_name.title().replace('_', '')}Test"
    _Test.__qualname__ = _Test.__name__
    return _Test


_REDUCTION_CONFIGS_6 = [
    {"shape": (16,), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": True, "dtype": torch.float32},
    {"shape": (2, 3, 4), "dim": 1, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": None, "keepdim": False, "dtype": torch.float32},
]

_REDUCTION_CONFIGS_5 = [
    {"shape": (16,), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": True, "dtype": torch.float32},
    {
        "shape": (4, 4),
        "dim": -1,
        "keepdim": False,
        "dtype": torch.float32,
        "correction": 0,
    },
    {"shape": (2, 3, 4), "dim": 1, "keepdim": False, "dtype": torch.float32},
]

_PROD_CONFIGS = [
    {"shape": (8,), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": True, "dtype": torch.float32},
    {"shape": (2, 3, 4), "dim": 1, "keepdim": False, "dtype": torch.float32},
]

_LOGSUMEXP_CONFIGS = [
    {"shape": (16,), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": 0, "keepdim": False, "dtype": torch.float32},
    {"shape": (4, 4), "dim": -1, "keepdim": True, "dtype": torch.float32},
    {"shape": (2, 3, 4), "dim": 1, "keepdim": False, "dtype": torch.float32},
]

# fmt: off
_REDUCTION_OP_TESTS = [
    {"op_name": "sum",       "op_fn": torch.sum,       "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "mean",      "op_fn": torch.mean,      "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "amax",      "op_fn": torch.amax,      "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "amin",      "op_fn": torch.amin,      "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "argmax",    "op_fn": torch.argmax,    "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "argmin",    "op_fn": torch.argmin,    "configs": _REDUCTION_CONFIGS_6},
    {"op_name": "prod",      "op_fn": torch.prod,      "configs": _PROD_CONFIGS, "input_fn": _input_fn(scale=0.5, offset=1.0)},
    {"op_name": "var",       "op_fn": torch.var,       "configs": _REDUCTION_CONFIGS_5, "has_correction": True},
    {"op_name": "std",       "op_fn": torch.std,       "configs": _REDUCTION_CONFIGS_5, "has_correction": True},
    {"op_name": "logsumexp", "op_fn": torch.logsumexp, "configs": _LOGSUMEXP_CONFIGS},
]
# fmt: on

for _entry in _REDUCTION_OP_TESTS:
    _cls = _make_reduction_op_test(**_entry)
    register_test(_cls)
    globals()[_cls.__name__] = _cls


# =============================================================================
# TRIANGULAR OPS
# =============================================================================


class TriangularModel(nn.Module):
    def __init__(self, mode: str = "tril", diagonal: int = 0):
        super().__init__()
        self.op_fn = torch.tril if mode == "tril" else torch.triu
        self.diagonal = diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op_fn(x, diagonal=self.diagonal)


_TRIANGULAR_CONFIGS = [
    {"shape": (4, 4), "diagonal": 0, "dtype": torch.float32},
    {"shape": (8, 8), "diagonal": 0, "dtype": torch.float32},
    {"shape": (4, 6), "diagonal": 0, "dtype": torch.float32},
    {"shape": (6, 4), "diagonal": 0, "dtype": torch.float32},
    {"shape": (4, 4), "diagonal": 1, "dtype": torch.float32},
    {"shape": (4, 4), "diagonal": -1, "dtype": torch.float32},
    {"shape": (4, 4), "diagonal": 2, "dtype": torch.float32},
    {"shape": (4, 4), "diagonal": 0, "dtype": torch.bfloat16},
    {"shape": (2, 4, 4), "diagonal": 0, "dtype": torch.float32},
    {"shape": (2, 3, 4, 4), "diagonal": 0, "dtype": torch.float32},
]


def _make_triangular_test(mode: str) -> type:
    """Generate a registered OpTestCase subclass for tril or triu."""

    class _Test(OpTestCase):
        name = mode

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
            self.name = f"{mode}_{shape_str}_{dtype_str}{diag_str}"

        @classmethod
        def get_test_configs(cls) -> List["_Test"]:
            return [cls(**c) for c in _TRIANGULAR_CONFIGS]

        def create_inputs(self) -> Tuple[torch.Tensor, ...]:
            return (torch.randn(self.shape, dtype=self.dtype),)

        def create_model(self) -> nn.Module:
            return TriangularModel(mode=mode, diagonal=self.diagonal)

    _Test.__name__ = f"{mode.title()}Test"
    _Test.__qualname__ = _Test.__name__
    return _Test


TrilTest = _make_triangular_test("tril")
TriuTest = _make_triangular_test("triu")
register_test(TrilTest)
register_test(TriuTest)


# =============================================================================
# LIKE OPS
# =============================================================================


class ZerosLikeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class OnesLikeModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class FullLikeModel(nn.Module):
    def __init__(self, fill_value: float, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.fill_value = fill_value
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.full_like(x, self.fill_value, dtype=self.dtype)
        if self.dtype is not None and self.dtype != x.dtype:
            return x * t.to(x.dtype)
        return t


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
        fill_dtype: Optional[torch.dtype] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        self.fill_dtype = fill_dtype
        if rtol is not None:
            self.rtol = rtol
        if atol is not None:
            self.atol = atol
        shape_str = "x".join(str(s) for s in shape)
        dtype_str = str(dtype).replace("torch.", "")
        fill_dtype_str = (
            f"_as_{str(fill_dtype).replace('torch.', '')}" if fill_dtype else ""
        )
        self.name = f"full_like_{shape_str}_v{fill_value}_{dtype_str}{fill_dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["FullLikeTest"]:
        return [
            cls(shape=(16,), fill_value=3.14, dtype=torch.float32),
            cls(shape=(4, 4), fill_value=2.71, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=-1.0, dtype=torch.float32),
            cls(shape=(4, 4), fill_value=0.5, dtype=torch.bfloat16),
            # Explicit fill_dtype exercises scalar_type serialization (optional_int).
            # 1.005859375 rounds differently in bf16 vs f32, so the model multiplies
            # the bf16 mask back into the f32 input to make the precision loss observable.
            cls(
                shape=(4, 4),
                fill_value=1.005859375,
                fill_dtype=torch.bfloat16,
                rtol=0.0,
                atol=0.0,
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.fill_dtype is not None:
            torch.manual_seed(42)
            return (torch.randn(self.shape, dtype=self.dtype) * 100,)
        return (torch.randn(self.shape, dtype=self.dtype),)

    def create_model(self) -> nn.Module:
        return FullLikeModel(fill_value=self.fill_value, dtype=self.fill_dtype)


# =============================================================================
# FULL, ZEROS, ONES OPS
# =============================================================================


class FullModel(nn.Module):
    def __init__(self, shape: Tuple[int, ...], fill_value: float, dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(self.shape, self.fill_value, dtype=self.dtype)


class ZerosModel(nn.Module):
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.shape, dtype=self.dtype)


class OnesModel(nn.Module):
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(self.shape, dtype=self.dtype)


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
        return [
            cls(shape=(2, 3, 4), fill_value=1.5, dtype=torch.float32),
            cls(shape=(10,), fill_value=0.0, dtype=torch.float32),
            cls(shape=(1, 128), fill_value=-2.5, dtype=torch.float32),
            cls(shape=(4, 8, 16), fill_value=3.14159, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=1.0, dtype=torch.bfloat16),
            cls(shape=(8, 16), fill_value=-1.0, dtype=torch.bfloat16),
            cls(shape=(2, 3, 4), fill_value=2.0, dtype=torch.float16),
            # Integer fill values (matching individual test file)
            cls(shape=(2, 3, 4), fill_value=0.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=1.0, dtype=torch.float32),
            cls(shape=(2, 3, 4), fill_value=-1.0, dtype=torch.float32),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(1, dtype=torch.float32)
        return (x,)

    def create_model(self) -> nn.Module:
        return FullModel(self.shape, self.fill_value, self.dtype)


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
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
            cls(shape=(1, 128), dtype=torch.float32),
            cls(shape=(4, 8, 16), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.bfloat16),
            cls(shape=(8, 16), dtype=torch.bfloat16),
            cls(shape=(2, 3, 4), dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
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
        return [
            cls(shape=(2, 3, 4), dtype=torch.float32),
            cls(shape=(10,), dtype=torch.float32),
            cls(shape=(1, 128), dtype=torch.float32),
            cls(shape=(4, 8, 16), dtype=torch.float32),
            cls(shape=(2, 3, 4), dtype=torch.bfloat16),
            cls(shape=(8, 16), dtype=torch.bfloat16),
            cls(shape=(2, 3, 4), dtype=torch.float16),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(1, dtype=torch.float32)
        return (x,)

    def create_model(self) -> nn.Module:
        return OnesModel(self.shape, self.dtype)


# =============================================================================
# TO_DTYPE OP
# =============================================================================


class ToDtypeModel(nn.Module):
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
        return [
            cls(
                shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.bfloat16
            ),
            cls(shape=(10,), source_dtype=torch.float32, target_dtype=torch.bfloat16),
            cls(
                shape=(1, 128), source_dtype=torch.float32, target_dtype=torch.bfloat16
            ),
            cls(
                shape=(2, 3, 4), source_dtype=torch.bfloat16, target_dtype=torch.float32
            ),
            cls(
                shape=(4, 8, 16),
                source_dtype=torch.bfloat16,
                target_dtype=torch.float32,
            ),
            cls(
                shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.float16
            ),
            cls(
                shape=(2, 3, 4), source_dtype=torch.float16, target_dtype=torch.float32
            ),
            cls(shape=(2, 3, 4), source_dtype=torch.float32, target_dtype=torch.int32),
            cls(shape=(2, 3, 4), source_dtype=torch.int32, target_dtype=torch.float32),
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


# =============================================================================
# BATCH NORM OPS
# =============================================================================


class BatchNormModel(nn.Module):
    def __init__(self, num_features: int, dtype: torch.dtype, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=affine, dtype=dtype)
        self.bn.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BatchNorm1dModel(nn.Module):
    def __init__(self, num_features: int, dtype: torch.dtype, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=affine, dtype=dtype)
        self.bn.eval()

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
        affine: bool = True,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.height = height
        self.width = width
        self.dtype = dtype
        self.affine = affine
        dtype_str = str(dtype).replace("torch.", "")
        prefix = "batch_norm_2d_no_affine" if not affine else "batch_norm_2d"
        self.name = f"{prefix}_{batch_size}x{num_features}x{height}x{width}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm2dTest"]:
        return [
            cls(batch_size=1, num_features=16, height=8, width=8, dtype=torch.float32),
            cls(
                batch_size=2, num_features=32, height=16, width=16, dtype=torch.float32
            ),
            cls(batch_size=4, num_features=64, height=4, width=4, dtype=torch.float32),
            cls(batch_size=2, num_features=16, height=8, width=8, dtype=torch.bfloat16),
            cls(batch_size=1, num_features=32, height=4, width=4, dtype=torch.bfloat16),
            cls(batch_size=2, num_features=16, height=8, width=8, dtype=torch.float16),
            # No-affine variants (no weight/bias)
            cls(
                batch_size=1,
                num_features=16,
                height=8,
                width=8,
                dtype=torch.float32,
                affine=False,
            ),
            cls(
                batch_size=2,
                num_features=32,
                height=4,
                width=4,
                dtype=torch.bfloat16,
                affine=False,
            ),
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
        return BatchNormModel(self.num_features, self.dtype, affine=self.affine)


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
        affine: bool = True,
    ):
        self.batch_size = batch_size
        self.num_features = num_features
        self.seq_len = seq_len
        self.dtype = dtype
        self.affine = affine
        dtype_str = str(dtype).replace("torch.", "")
        prefix = "batch_norm_1d_no_affine" if not affine else "batch_norm_1d"
        self.name = f"{prefix}_{batch_size}x{num_features}x{seq_len}_{dtype_str}"

    @classmethod
    def get_test_configs(cls) -> List["BatchNorm1dTest"]:
        return [
            cls(batch_size=1, num_features=16, seq_len=32, dtype=torch.float32),
            cls(batch_size=2, num_features=32, seq_len=64, dtype=torch.float32),
            cls(batch_size=2, num_features=16, seq_len=32, dtype=torch.bfloat16),
            cls(batch_size=2, num_features=16, seq_len=32, dtype=torch.float16),
            # No-affine variants (no weight/bias)
            cls(
                batch_size=1,
                num_features=16,
                seq_len=32,
                dtype=torch.float32,
                affine=False,
            ),
            cls(
                batch_size=2,
                num_features=32,
                seq_len=64,
                dtype=torch.bfloat16,
                affine=False,
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.num_features, self.seq_len, dtype=self.dtype
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return BatchNorm1dModel(self.num_features, self.dtype, affine=self.affine)


# =============================================================================
# ATTENTION OPS (RMSNorm, SDPA, RoPE)
# Note: These tests require additional imports that are loaded conditionally
# =============================================================================


class SDPAModel(nn.Module):
    """Basic scaled dot product attention."""

    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class SDPAWithMaskModel(nn.Module):
    """SDPA with explicit attention mask (additive float format)."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class SDPAWithBoolMaskModel(nn.Module):
    """SDPA with boolean attention mask.

    This tests the case where a boolean mask is passed to SDPA.
    PyTorch expects: True = attend, False = masked out.
    """

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class GQAModel(nn.Module):
    """Grouped Query Attention - fewer KV heads than Q heads."""

    def __init__(self, num_heads: int, num_kv_heads: int, is_causal: bool = False):
        super().__init__()
        self.num_groups = num_heads // num_kv_heads
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


@register_test
class SDPATest(OpTestCase):
    """Test case for SDPA."""

    name = "sdpa"
    rtol = 1e-3
    atol = 1e-3
    expected_node_counts = {"SdpaNode": 1, "ExpandDimsNode": 0}

    def __init__(
        self,
        batch_size: int = 2,
        num_heads: int = 8,
        seq_len: int = 32,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        is_causal: bool = False,
        use_mask: bool = False,
        use_bool_mask: bool = False,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.is_causal = is_causal
        self.use_mask = use_mask
        self.use_bool_mask = use_bool_mask

        parts = ["sdpa"]
        if num_kv_heads is not None:
            parts.append(f"gqa{num_kv_heads}")
        if is_causal:
            parts.append("causal")
        if use_mask:
            parts.append("mask")
        if use_bool_mask:
            parts.append("bool_mask")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["SDPATest"]:
        return [
            cls(),
            cls(is_causal=True),
            cls(num_kv_heads=4),
            cls(use_mask=True),
            cls(use_bool_mask=True),  # Test boolean mask conversion
        ]

    def create_model(self) -> nn.Module:
        if self.use_mask:
            return SDPAWithMaskModel()
        elif self.use_bool_mask:
            return SDPAWithBoolMaskModel()
        elif self.num_kv_heads is not None:
            return GQAModel(self.num_heads, self.num_kv_heads, self.is_causal)
        else:
            return SDPAModel(self.is_causal)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        kv_heads = self.num_kv_heads if self.num_kv_heads else self.num_heads
        k = torch.randn(self.batch_size, kv_heads, self.seq_len, self.head_dim)
        v = torch.randn(self.batch_size, kv_heads, self.seq_len, self.head_dim)

        if self.use_mask:
            # Additive float mask: 0 = attend, -inf = masked
            mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
            mask[:, :, :, : self.seq_len // 4] = float("-inf")
            return (q, k, v, mask)
        elif self.use_bool_mask:
            # Boolean mask: True = attend, False = masked
            # This tests that the backend correctly converts bool -> additive format
            mask = torch.ones(
                self.batch_size, 1, self.seq_len, self.seq_len, dtype=torch.bool
            )
            mask[:, :, :, : self.seq_len // 4] = False  # Mask out first quarter
            return (q, k, v, mask)
        return (q, k, v)


# =============================================================================
# CUSTOM SDPA OP (mlx::custom_sdpa)
# =============================================================================


class CustomSDPAModel(nn.Module):
    """
    Test model for mlx::custom_sdpa with ETKVCache.

    Simulates a single attention layer: updates the KV cache, then calls
    mlx::custom_sdpa which slices K/V to [0:start_pos+seq_len] and runs SDPA.
    """

    def __init__(
        self,
        max_context_length: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache = ETKVCache(
            max_batch_size=1,
            max_context_length=max_context_length,
            n_heads=n_kv_heads,
            head_dim=head_dim,
            enable_dynamic_shape=True,
        )

    def forward(
        self,
        input_pos: torch.Tensor,  # [S] position indices
        q: torch.Tensor,  # [B, n_heads, S, D]
        k_val: torch.Tensor,  # [B, n_kv_heads, S, D]
        v_val: torch.Tensor,  # [B, n_kv_heads, S, D]
    ) -> torch.Tensor:
        # Update KV cache and get full cache tensors
        k_cache, v_cache = self.cache.update(input_pos, k_val, v_val)

        start_pos = input_pos[0].item()

        output = torch.ops.mlx.custom_sdpa(
            q,
            k_cache,
            v_cache,
            start_pos=start_pos,
            is_causal=True,
            scale=self.head_dim**-0.5,
        )
        return output


@register_test
class CustomSDPATest(OpTestCase):
    """
    Test case for mlx::custom_sdpa with KV cache slicing.

    Verifies that custom_sdpa:
    1. Correctly slices K/V cache to [0:start_pos+seq_len]
    2. Produces numerically correct attention output
    3. Handles GQA (fewer KV heads than Q heads)
    4. Works with dynamic shapes (varying seq_len and start_pos)
    """

    name = "custom_sdpa"
    rtol = 1e-3
    atol = 1e-3
    expected_node_counts = {
        "SdpaNode": 1,
        "SliceUpdateNode": 2,
        "SliceNode": 2,
        "IdCopyNode": 2,
        "ExpandDimsNode": 0,
    }

    def __init__(
        self,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        head_dim: int = 64,
        max_context_length: int = 128,
        seq_len: int = 8,
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.seq_len = seq_len

        parts = ["custom_sdpa"]
        if n_kv_heads != n_heads:
            parts.append(f"gqa{n_kv_heads}")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["CustomSDPATest"]:
        return [
            cls(),  # MHA
            cls(n_kv_heads=4),  # GQA (8 Q heads, 4 KV heads)
            cls(n_kv_heads=1),  # MQA (8 Q heads, 1 KV head)
        ]

    def create_model(self) -> nn.Module:
        return CustomSDPAModel(
            max_context_length=self.max_context_length,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        input_pos = torch.tensor([0], dtype=torch.int64)
        q = torch.randn(1, self.n_heads, self.seq_len, self.head_dim)
        k = torch.randn(1, self.n_kv_heads, self.seq_len, self.head_dim)
        v = torch.randn(1, self.n_kv_heads, self.seq_len, self.head_dim)
        return (input_pos, q, k, v)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        test_seq_len = self.seq_len + 4
        input_pos = torch.tensor([16], dtype=torch.int64)
        q = torch.randn(1, self.n_heads, test_seq_len, self.head_dim)
        k = torch.randn(1, self.n_kv_heads, test_seq_len, self.head_dim)
        v = torch.randn(1, self.n_kv_heads, test_seq_len, self.head_dim)
        return (input_pos, q, k, v)

    def get_dynamic_shapes(self) -> Optional[Dict[str, any]]:
        seq_dim = Dim("seq_len", min=1, max=self.max_context_length)
        return {
            "input_pos": None,
            "q": {2: seq_dim},
            "k_val": {2: seq_dim},
            "v_val": {2: seq_dim},
        }


# =============================================================================
# QUANTIZED OPS
# Note: These require TorchAO to be installed
# =============================================================================


class QuantizedLinearModel(nn.Module):
    """Simple linear layer that will be quantized."""

    def __init__(
        self, in_features: int = 64, out_features: int = 128, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@register_test
class QuantizedLinearTest(OpTestCase):
    """Test case for TorchAO int4 quantized nn.Linear."""

    name = "quantized_linear"
    rtol = 0.1
    atol = 0.1

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 128,
        batch_size: int = 2,
        seq_len: int = 16,
        bias: bool = True,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bias = bias
        self.group_size = group_size
        self.dtype = dtype

        parts = ["quantized_linear", f"g{group_size}"]
        if not bias:
            parts.append("no_bias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["QuantizedLinearTest"]:
        return [
            cls(),
        ]

    def create_model(self) -> nn.Module:
        model = QuantizedLinearModel(
            self.in_features, self.out_features, bias=self.bias
        )
        model = model.to(self.dtype)

        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                ),
            )
        except ImportError:
            raise RuntimeError("TorchAO not installed. Run: pip install torchao")

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.in_features, dtype=self.dtype
        )
        return (x,)


class QuantizedEmbeddingModel(nn.Module):
    """Simple embedding layer that will be quantized."""

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


@register_test
class QuantizedEmbeddingTest(OpTestCase):
    """Test case for TorchAO int4 quantized nn.Embedding."""

    name = "quantized_embedding"
    rtol = 0.1
    atol = 0.1

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 64,
        batch_size: int = 2,
        seq_len: int = 16,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.group_size = group_size
        self.dtype = dtype

        parts = ["quantized_embedding", f"g{group_size}"]
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["QuantizedEmbeddingTest"]:
        return [
            cls(),
        ]

    def create_model(self) -> nn.Module:
        model = QuantizedEmbeddingModel(self.num_embeddings, self.embedding_dim)
        model = model.to(self.dtype)

        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            def embedding_filter(module: nn.Module, fqn: str) -> bool:
                return isinstance(module, nn.Embedding)

            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                ),
                embedding_filter,
            )
        except ImportError:
            raise RuntimeError("TorchAO not installed. Run: pip install torchao")

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randint(0, self.num_embeddings, (self.batch_size, self.seq_len))
        return (x,)


class DequantizeConv2dModel(nn.Module):
    """Conv2d layer whose weight will be quantized.

    The pattern matcher only fuses dequantize_affine with linear and embedding.
    A quantized Conv2d produces a standalone dequantize_affine node in the graph,
    exercising the DequantizeNode path.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class DequantizeTest(OpTestCase):
    """Test case for standalone TorchAO dequantize_affine (DequantizeNode).

    Uses a quantized Conv2d to produce a standalone dequantize_affine node,
    since the pattern matcher only fuses dequantize with linear/embedding.
    """

    name = "dequantize"
    rtol = 0.1
    atol = 0.1

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        height: int = 8,
        width: int = 8,
        batch_size: int = 1,
        group_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.group_size = group_size
        self.dtype = dtype

        parts = ["dequantize", f"g{group_size}"]
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["DequantizeTest"]:
        return [
            cls(),
        ]

    def create_model(self) -> nn.Module:
        model = DequantizeConv2dModel(
            self.in_channels, self.out_channels, self.kernel_size
        )
        model = model.to(self.dtype)

        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

        def conv2d_filter(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Conv2d)

        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
            ),
            conv2d_filter,
        )

        return model

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.height,
            self.width,
            dtype=self.dtype,
        )
        return (x,)
