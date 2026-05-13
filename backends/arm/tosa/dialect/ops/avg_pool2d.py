# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
    TosaSpecification,
)
from torch.fx.experimental.symbolic_shapes import FloorDiv
from torch.types import IntLikeType


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


def _get_supported_avg_pool2d_acc_types(
    tosa_spec: TosaSpecification,
) -> dict[torch.dtype, tuple[torch.dtype, ...]]:
    supported_acc_types: dict[torch.dtype, tuple[torch.dtype, ...]] = {}

    if tosa_spec.support_integer():
        supported_acc_types[torch.int8] = (torch.int32,)
        if tosa_spec.support_extension("int16"):
            supported_acc_types[torch.int16] = (torch.int32,)

    if tosa_spec.support_float():
        supported_acc_types[torch.float16] = (torch.float16, torch.float32)
        supported_acc_types[torch.float32] = (torch.float32,)
        if tosa_spec.support_extension("bf16"):
            supported_acc_types[torch.bfloat16] = (torch.float32,)

    return supported_acc_types


def validate_avg_pool2d_dtype(
    tosa_spec: TosaSpecification,
    x: torch.Tensor,
    input_zp: torch.Tensor,
    output_zp: torch.Tensor,
    acc_type: torch.dtype,
    op: str,
) -> None:
    """Validate dtypes for TOSA AVG_POOL2D and AVG_POOL2D_ADAPTIVE."""
    supported_acc_types = _get_supported_avg_pool2d_acc_types(tosa_spec)
    if x.dtype not in supported_acc_types:
        raise TosaValueError(
            f"Unsupported input dtype {x.dtype}, supported types are {tuple(supported_acc_types)}",
            op=op,
        )

    if input_zp.dtype != x.dtype:
        raise TosaValueError(
            f"{op} requires input zero-point dtype {input_zp.dtype} to match input dtype {x.dtype}",
            op=op,
        )
    if output_zp.dtype != x.dtype:
        raise TosaValueError(
            f"{op} requires output zero-point dtype {output_zp.dtype} to match input dtype {x.dtype}",
            op=op,
        )

    valid_acc_types = supported_acc_types[x.dtype]
    if acc_type not in valid_acc_types:
        raise TosaValueError(
            f"{op} accumulator type must be one of {valid_acc_types}, got {acc_type}",
            op=op,
        )


def validate_avg_pool2d_args(
    kernel: List[IntLikeType] | List[int],
    stride: List[IntLikeType] | List[int],
    pad: List[IntLikeType] | List[int],
    op: str,
) -> None:
    if len(kernel) != 2 or len(stride) != 2 or len(pad) != 4:
        raise TosaValueError(
            f"{op} expects kernel of length 2, stride of length 2, pad of length 4; got "
            f"kernel={kernel}, stride={stride}, pad={pad}",
            op=op,
        )


@register_fake_tosa_op(
    "AVG_POOL2D(Tensor input, Tensor input_zp, Tensor output_zp, int[2] kernel, int[2] stride, SymInt[4] pad, ScalarType acc_type) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def AVG_POOL2D(
    x: torch.Tensor,
    input_zp: torch.Tensor,
    output_zp: torch.Tensor,
    kernel: List[int],
    stride: List[int],
    pad: List[Union[int, torch.SymInt]],
    acc_type: torch.dtype,
) -> torch.Tensor:
    """Compute output meta for a TOSA AVG_POOL2D operation."""
    tosa_spec = get_context_spec()
    validate_avg_pool2d_dtype(
        tosa_spec, x, input_zp, output_zp, acc_type, op="AVG_POOL2D"
    )
    output_shape = compute_avg_pool2d_output_shape(
        x,
        kernel,
        stride,
        pad,
        op="AVG_POOL2D",
    )
    return torch.empty(size=output_shape, dtype=x.dtype)


def compute_avg_pool2d_output_shape(
    x: torch.Tensor,
    kernel: List[IntLikeType] | List[int],
    stride: List[IntLikeType] | List[int],
    pad: List[IntLikeType] | List[int],
    op: str = "AVG_POOL2D",
) -> List[IntLikeType]:
    """Compute the output shape for NHWC avg-pool."""

    if x.dim() != 4:
        raise TosaValueError(f"{op} requires a 4D tensor, got {x.dim()}D", op=op)

    validate_avg_pool2d_args(kernel, stride, pad, op=op)

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
