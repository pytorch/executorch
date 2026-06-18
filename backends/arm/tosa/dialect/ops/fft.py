# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
    TosaSpecification,
)
from torch.utils._sympy.functions import FloorDiv


def _validate_fft_spec(op: str) -> None:
    tosa_spec = get_context_spec()
    if not (tosa_spec.support_float() and tosa_spec.support_extension("fft")):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support {op}",
            op=op,
        )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_power_of_two(value: int | torch.SymInt, name: str, op: str) -> None:
    if isinstance(value, torch.SymInt):
        expr = sympy.simplify(_to_sympy_expr(value))
        value_range = get_context_shape_env().bound_sympy(expr)
        if value_range.is_int and value_range.is_singleton():
            singleton = sympy.simplify(value_range.lower)
            if singleton.is_integer and not _is_power_of_two(int(singleton)):
                raise TosaValueError(
                    f"{op} requires {name} to be a power of two but got {singleton}",
                    op=op,
                )
        return

    if not _is_power_of_two(int(value)):
        raise TosaValueError(
            f"{op} requires {name} to be a power of two but got {value}",
            op=op,
        )


def _validate_fft_input(input_real: torch.Tensor, op: str) -> None:
    if input_real.dtype != torch.float32:
        raise TosaValueError(f"{op} requires float32 inputs", op=op)
    if input_real.dim() != 3:
        raise TosaValueError(f"{op} requires a rank-3 input", op=op)

    _, height, width = input_real.shape
    _validate_power_of_two(height, "height", op)
    _validate_power_of_two(width, "width", op)


def _to_sympy_expr(value: int | torch.SymInt) -> sympy.Expr:
    if isinstance(value, torch.SymInt):
        return value.node._expr
    return sympy.Integer(int(value))


def _rfft_output_width(width: int | torch.SymInt) -> int | torch.SymInt:
    if isinstance(width, torch.SymInt):
        expr = FloorDiv(_to_sympy_expr(width), sympy.Integer(2)) + sympy.Integer(1)
        return get_context_shape_env().create_symintnode(expr, hint=None)
    return width // 2 + 1


def _same_fft_dimension(lhs: int | torch.SymInt, rhs: int | torch.SymInt) -> bool:
    if not isinstance(lhs, torch.SymInt) and not isinstance(rhs, torch.SymInt):
        return lhs == rhs

    diff = sympy.simplify(_to_sympy_expr(lhs) - _to_sympy_expr(rhs))
    if diff == 0:
        return True

    value_range = get_context_shape_env().bound_sympy(diff)
    return (
        value_range.is_int
        and value_range.is_singleton()
        and sympy.simplify(value_range.lower) == 0
    )


def _same_fft_shape(
    lhs: torch.Size | tuple[int | torch.SymInt, ...],
    rhs: torch.Size | tuple[int | torch.SymInt, ...],
) -> bool:
    return len(lhs) == len(rhs) and all(
        _same_fft_dimension(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs, rhs)
    )


@register_tosa_op(
    "FFT2D(Tensor input_real, Tensor input_imag, *, bool inverse=False, bool local_bound=False) -> (Tensor output_real, Tensor output_imag)",
    TosaSpecification.all_versions_and_profiles(),
)
def FFT2D(
    input_real: torch.Tensor,
    input_imag: torch.Tensor,
    *,
    inverse: bool = False,
    local_bound: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_fft_spec("FFT2D")
    _validate_fft_input(input_real, "FFT2D")
    _validate_fft_input(input_imag, "FFT2D")

    if not _same_fft_shape(input_real.shape, input_imag.shape):
        raise TosaValueError(
            f"FFT2D expects matching input shapes but got {tuple(input_real.shape)} and {tuple(input_imag.shape)}",
            op="FFT2D",
        )

    return (
        torch.empty_like(input_real, dtype=input_real.dtype),
        torch.empty_like(input_imag, dtype=input_imag.dtype),
    )


@register_tosa_op(
    "RFFT2D(Tensor input_real, *, bool local_bound=False) -> (Tensor output_real, Tensor output_imag)",
    TosaSpecification.all_versions_and_profiles(),
)
def RFFT2D(
    input_real: torch.Tensor,
    *,
    local_bound: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_fft_spec("RFFT2D")
    _validate_fft_input(input_real, "RFFT2D")

    batch, height, width = input_real.shape
    output_shape = (batch, height, _rfft_output_width(width))
    return (
        torch.empty(output_shape, dtype=input_real.dtype),
        torch.empty(output_shape, dtype=input_real.dtype),
    )
