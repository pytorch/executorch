# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
)
from torch.fx.experimental.symbolic_shapes import FloorDiv
from torch.types import IntLikeType

_VALID_NAN_MODES = {"PROPAGATE", "IGNORE"}


def accepts_int64(dtype: torch.dtype, op: str) -> bool:
    """Whether `dtype` is an int64 the active specification carries."""
    if dtype != torch.int64:
        return False
    if not get_context_spec().support_extension("int64"):
        raise TosaValueError(
            f"TOSA spec {get_context_spec()} doesn't support "
            f"{str(dtype).removeprefix('torch.')} for {op}",
            op=op,
        )
    return True


def broadcast_shape(
    input1: torch.Tensor, input2: torch.Tensor, op: str
) -> tuple[int | torch.SymInt, ...]:
    try:
        return tuple(torch.broadcast_shapes(input1.shape, input2.shape))
    except (RuntimeError, ValueError) as err:
        raise TosaValueError(
            f"Failed to broadcast shapes {tuple(input1.shape)} and {tuple(input2.shape)}",
            op=op,
        ) from err


def require_same_dtype(input1: torch.Tensor, input2: torch.Tensor, op: str) -> None:
    if input1.dtype != input2.dtype:
        raise TosaValueError(
            f"Expected matching dtypes but got {input1.dtype} and {input2.dtype}",
            op=op,
        )


def binary_meta(
    input1: torch.Tensor,
    input2: torch.Tensor,
    op: str,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    require_same_dtype(input1, input2, op)
    output_shape = broadcast_shape(input1, input2, op)
    return torch.empty(output_shape, dtype=output_dtype or input1.dtype)


def validate_float_dtype(dtype: torch.dtype, op: str) -> None:
    tosa_spec = get_context_spec()

    if dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support {dtype} for {op}",
                op=op,
            )
        return

    if dtype == torch.bfloat16:
        if not (tosa_spec.support_float() and tosa_spec.support_extension("bf16")):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support bfloat16 for {op}",
                op=op,
            )
        return

    raise TosaValueError(f"Unsupported dtype {dtype} for {op}", op=op)


def validate_nan_mode(nan_mode: str, op: str) -> None:
    if nan_mode not in _VALID_NAN_MODES:
        raise TosaValueError(
            f"Unsupported nan_mode {nan_mode}. Expected one of {_VALID_NAN_MODES}",
            op=op,
        )


def validate_power_of_two(size: int | torch.SymInt, name: str, op: str) -> None:
    if not isinstance(size, int):
        return
    if size < 1 or (size & (size - 1)) != 0:
        raise TosaValueError(
            f"{name} must be a positive power of two, got {size}", op=op
        )


def _to_sympy_expr(value: int | torch.SymInt) -> sympy.Expr:
    if isinstance(value, torch.SymInt):
        return value.node._expr
    return sympy.Integer(int(value))


def _from_sympy_expr(expr: sympy.Expr) -> int | torch.SymInt:
    # Full `sympy.simplify()` is too expensive for the large symbolic formulas
    # produced by dynamic-shape model lowering. Keep the expression in its raw
    # symbolic form and only fold obviously-static integers.
    if expr.is_Integer:
        return int(expr)
    return get_context_shape_env().create_symintnode(expr, hint=None)


def compute_pool2d_output_shape(
    x: torch.Tensor,
    kernel: list[IntLikeType] | list[int],
    stride: list[IntLikeType] | list[int],
    pad: list[IntLikeType] | list[int],
) -> list[IntLikeType]:
    """Compute the output shape for NHWC pooling."""
    n, h, w, c = x.shape
    k_h, k_w = kernel
    s_h, s_w = stride
    p_top, p_bot, p_left, p_right = pad

    h_expr = (
        FloorDiv(
            _to_sympy_expr(h) + _to_sympy_expr(p_top) + _to_sympy_expr(p_bot) - k_h,
            s_h,
        )
        + 1
    )
    w_expr = (
        FloorDiv(
            _to_sympy_expr(w) + _to_sympy_expr(p_left) + _to_sympy_expr(p_right) - k_w,
            s_w,
        )
        + 1
    )

    h_out = _from_sympy_expr(h_expr)
    w_out = _from_sympy_expr(w_expr)
    return [n, h_out, w_out, c]
