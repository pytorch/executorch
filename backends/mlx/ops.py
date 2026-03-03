#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Op Handlers - registered handlers for converting ATen/custom ops to MLX.

This module contains all the op handler functions registered with the MLXOpRegistry.
Each handler converts a specific PyTorch operation to the corresponding MLX graph node.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import AddmmNode
from torch.fx.node import Node


def require_static_int(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static integer (not a Slot/SymInt).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, int):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static int32 value, but got {value} (type={type(value).__name__})."
        )


def require_static_float(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static float (not a Slot/SymFloat).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, (int, float)):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static float value, but got {value} (type={type(value).__name__})."
        )


def require_static_ints(
    values: Union[List[Any], Any], param_name: str, op_name: str
) -> None:
    """
    Validate that all values in a list are static integers (not Slots/SymInts).

    Raises NotImplementedError if any value is dynamic.

    Args:
        values: List of values to check, or a single value
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if not isinstance(values, list):
        values = [values]

    for v in values:
        require_static_int(v, param_name, op_name)


def require_args(
    args: List[Any],
    min_count: int,
    max_count: int,
    op_name: str,
) -> None:
    """
    Validate that args count is within expected range.

    Raises ValueError if the count is outside the expected range.

    Args:
        args: The handler args list
        min_count: Minimum number of args expected
        max_count: Maximum number of args expected
        op_name: Name of the operation (for error message)
    """
    if not (min_count <= len(args) <= max_count):
        if min_count == max_count:
            raise ValueError(f"{op_name}: expected {min_count} args, got {len(args)}")
        raise ValueError(
            f"{op_name}: expected {min_count}-{max_count} args, got {len(args)}"
        )


def require_kwargs(
    kwargs: Dict[str, Any],
    allowed: Set[str],
    op_name: str,
) -> None:
    """
    Validate that only allowed kwargs are present.

    Raises ValueError if unexpected kwargs are found.

    Args:
        kwargs: The handler kwargs dict
        allowed: Set of allowed kwarg names
        op_name: Name of the operation (for error message)
    """
    unexpected = set(kwargs.keys()) - allowed
    if unexpected:
        raise ValueError(f"{op_name}: unexpected kwargs: {unexpected}")


def require_contiguous_format(
    *,
    layout=None,
    memory_format=None,
    dim_order=None,
    op_name: str,
) -> None:
    """
    Validate that layout/memory_format/dim_order specify contiguous format.

    MLX only supports contiguous (strided) tensors. Raises ValueError if
    sparse layouts or non-contiguous memory formats are requested.

    Args:
        layout: The torch layout (e.g., torch.strided, torch.sparse_coo)
        memory_format: The torch memory format (e.g., torch.contiguous_format,
            torch.channels_last)
        dim_order: The dimension order (list of ints, identity = contiguous)
        op_name: Name of the operation (for error message)
    """
    if layout is not None and layout != torch.strided:
        raise ValueError(f"{op_name}: only strided layout supported, got {layout}")

    if memory_format is not None and memory_format not in (
        torch.contiguous_format,
        torch.preserve_format,
    ):
        raise ValueError(
            f"{op_name}: only contiguous memory format supported, got {memory_format}"
        )

    if dim_order is not None:
        if list(dim_order) != list(range(len(dim_order))):
            raise ValueError(
                f"{op_name}: only contiguous dim_order supported, got {dim_order}"
            )


def is_static_value(value: Any) -> bool:
    """
    Check if a value is static (not a Slot/SymInt).

    Returns:
        True if the value is a static scalar (int, float, bool), False otherwise
    """
    return not isinstance(value, Slot)


def used_getitem_indices(n: Node) -> Set[int]:
    """Return the set of getitem indices actually consumed downstream.

    Only includes indices where the getitem node has at least one user.
    """
    return {
        user.args[1]
        for user in n.users
        if user.target == operator.getitem and len(user.users) > 0
    }


def normalize_reduction_dim(
    args: List[Any], start_idx: int = 1
) -> Tuple[Optional[List[int]], bool]:
    """
    Normalize dim argument for reduction operations.

    Extracts and normalizes the dim argument from handler args, returning a list of axes
    and the keepdim flag. Handles both list-based dims (e.g., sum.dim_IntList) and
    single int dims (e.g., prod.dim_int).

    Args:
        args: The handler args list
        start_idx: Index where the dim argument starts (default 1, after self)

    Returns:
        Tuple of (axes, keepdim) where:
        - axes: List of dimension indices, or empty list for reduce-all
        - keepdim: Boolean keepdim flag (default False)
    """
    if len(args) > start_idx and isinstance(args[start_idx], (list, tuple)):
        dim = list(args[start_idx])
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    elif len(args) > start_idx and isinstance(args[start_idx], int):
        dim = [args[start_idx]]
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    else:
        dim = []
        keepdim = False

    return dim, keepdim


<<<<<<< HEAD
@REGISTRY.register(target=["NOOP", torch.ops.aten._assert_scalar.default])
def _noop_handler(P: MLXProgramBuilder, n: Node) -> None:
    """No-op handler for nodes that don't emit any MLX instructions."""
    return None


=======
>>>>>>> e3b488076f (up)
@REGISTRY.register(target=[torch.ops.aten.addmm.default])
def _addmm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle addmm: self + (mat1 @ mat2).

    addmm(self, mat1, mat2, *, beta=1, alpha=1) computes:
        beta * self + alpha * (mat1 @ mat2)

    This is typically the result of decomposing linear(x, w, b) in Edge IR:
        permute(w) -> addmm(b, x, permuted_w)

    For the common case where beta=1 and alpha=1, this is equivalent to:
        mat1 @ mat2 + self

    We use AddmmNode which calls matmul directly (no transposition needed).
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 3, 3, "aten.addmm")
    require_kwargs(kwargs, {"beta", "alpha"}, "aten.addmm")
    bias, mat1, mat2 = args[0], args[1], args[2]

    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    out = P.make_or_get_slot(n)

    # Emit AddmmNode with alpha and beta parameters
    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(bias),
            alpha=float(alpha),
            beta=float(beta),
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
    ]
)
def _mm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle mm/bmm/matmul: matrix multiplication without bias.

    All three ops compute matrix products with different dimension expectations:
    - mm: 2D x 2D
    - bmm: 3D x 3D (batched)
    - matmul: arbitrary dimensions (NumPy semantics)

    MLX's matmul handles all cases, so we emit AddmmNode with bias=None.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.mm/bmm/matmul")
    require_kwargs(P.kwargs(n), set(), "aten.mm/bmm/matmul")
    mat1, mat2 = args[0], args[1]

    out = P.make_or_get_slot(n)

    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=None,
        )
    )
    return out
