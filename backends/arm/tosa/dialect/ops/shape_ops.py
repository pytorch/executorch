# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import sympy  # type: ignore[import-untyped]
import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op

from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
    TosaSpecification,
)
from torch.types import IntLikeType
from torch.utils._sympy.functions import FloorDiv


def _require_shape_extension(op: str) -> None:
    tosa_spec = get_context_spec()
    if not tosa_spec.support_extension("shape"):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support shape extension", op=op
        )


def _to_sympy_expr(value: IntLikeType) -> sympy.Expr:
    if isinstance(value, torch.SymInt):
        return value.node._expr
    return sympy.Integer(int(value))


def _to_lowest_concrete_int(value: IntLikeType, op: str, name: str) -> int:
    expr = _to_sympy_expr(value)
    if expr.is_integer is False:
        raise TosaValueError(f"{op} requires integer {name}", op=op)
    if expr.is_number:
        return int(expr)

    value_range = _get_expr_range(expr)
    if (
        value_range is not None
        and value_range.is_int
        and value_range.is_singleton()
        and value_range.lower.is_number
    ):
        return int(value_range.lower)

    raise TosaValueError(
        f"{op} requires compile-time constant {name}",
        op=op,
    )


def _require_known_nonnegative(value: IntLikeType, op: str, name: str) -> None:
    expr = _to_sympy_expr(value)
    if expr.is_number and int(expr) < 0:
        raise TosaValueError(f"{op} requires {name} >= 0", op=op)
    if expr.is_nonnegative is False:
        raise TosaValueError(f"{op} requires {name} >= 0", op=op)


def _require_known_positive(value: IntLikeType, op: str, name: str) -> None:
    expr = _to_sympy_expr(value)
    if expr.is_number and int(expr) < 1:
        raise TosaValueError(f"{op} requires {name} > 0", op=op)
    if expr.is_positive is False or expr.is_zero is True:
        raise TosaValueError(f"{op} requires {name} > 0", op=op)


def _require_known_less_than(
    value: IntLikeType, limit: int, op: str, name: str
) -> None:
    expr = _to_sympy_expr(value)
    if expr.is_number and int(expr) >= limit:
        raise TosaValueError(f"{op} requires {name} < {limit}", op=op)
    if sympy.Ge(expr, sympy.Integer(limit)) is sympy.true:
        raise TosaValueError(f"{op} requires {name} < {limit}", op=op)


def _get_expr_range(expr: sympy.Expr):
    try:
        shape_env = get_context_shape_env()
    except RuntimeError:
        return None

    try:
        return shape_env.bound_sympy(sympy.simplify(expr))
    except Exception:
        return None


def _is_definitely_value(expr: sympy.Expr, value: int) -> bool:
    if sympy.simplify(expr - value) == 0:
        return True

    value_range = _get_expr_range(expr)
    if value_range is None or not value_range.is_int or not value_range.is_singleton():
        return False

    lower = value_range.lower
    return lower.is_integer and lower.is_number and int(lower) == value


def _is_definitely_mismatch(lhs_expr: sympy.Expr, rhs_expr: sympy.Expr) -> bool:
    if lhs_expr.is_number and rhs_expr.is_number:
        return int(lhs_expr) != int(rhs_expr)

    if sympy.Ne(lhs_expr, rhs_expr) is sympy.true:
        return True

    lhs_range = _get_expr_range(lhs_expr)
    rhs_range = _get_expr_range(rhs_expr)
    if (
        lhs_range is None
        or rhs_range is None
        or not lhs_range.is_int
        or not rhs_range.is_int
    ):
        return False

    bounds = (
        lhs_range.lower,
        lhs_range.upper,
        rhs_range.lower,
        rhs_range.upper,
    )
    if not all(bound.is_number for bound in bounds):
        return False

    lhs_lower, lhs_upper, rhs_lower, rhs_upper = (int(bound) for bound in bounds)
    return lhs_upper < rhs_lower or rhs_upper < lhs_lower


def _to_finite_int_values(
    value: IntLikeType,
    op: str,
    name: str,
    *,
    max_values: int,
) -> list[int] | None:
    expr = _to_sympy_expr(value)
    if expr.is_integer is False:
        raise TosaValueError(f"{op} requires integer {name}", op=op)
    if expr.is_number:
        return [int(expr)]

    value_range = _get_expr_range(expr)
    if value_range is None or not value_range.is_int:
        return None

    lower = value_range.lower
    upper = value_range.upper
    if not lower.is_number or not upper.is_number:
        return None

    lower_i = int(lower)
    upper_i = int(upper)
    if upper_i < lower_i:
        return None

    num_values = upper_i - lower_i + 1
    if num_values > max_values:
        return None

    return list(range(lower_i, upper_i + 1))


def _supported_dim_dtypes(tosa_spec: TosaSpecification) -> list[torch.dtype]:
    supported = [torch.bool]
    if tosa_spec.support_integer():
        supported.extend([torch.int8, torch.int16, torch.int32])
    if tosa_spec.support_float():
        supported.extend([torch.float16, torch.float32])
    if tosa_spec.support_extension("bf16"):
        supported.append(torch.bfloat16)
    if tosa_spec.support_extension("int64"):
        supported.append(torch.int64)
    if tosa_spec.support_extension("fp8e4m3"):
        supported.append(torch.float8_e4m3fn)
    if tosa_spec.support_extension("fp8e5m2"):
        supported.append(torch.float8_e5m2)
    return supported


def _combine_shapes(
    lhs: list[IntLikeType],
    rhs: list[IntLikeType],
    combine: Callable[[sympy.Expr, sympy.Expr], sympy.Expr | sympy.Integer],
) -> list[IntLikeType]:
    if len(lhs) != len(rhs):
        raise ValueError(
            f"Expected shapes to be of same length, got {len(lhs)} and {len(rhs)}"
        )

    expr_lhs = [_to_sympy_expr(v) for v in lhs]
    expr_rhs = [_to_sympy_expr(v) for v in rhs]

    result: list[IntLikeType] = []
    for a, b in zip(expr_lhs, expr_rhs):
        expr = combine(a, b)
        if expr.is_number and expr.is_integer:
            result.append(int(expr))
            continue

        shape_env = get_context_shape_env()
        result.append(shape_env.create_symintnode(expr, hint=None))
    return result


@register_fake_tosa_op(
    "ADD_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def ADD_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("ADD_SHAPE")
    return _combine_shapes(shape1, shape2, lambda a, b: a + b)


@register_fake_tosa_op(
    "ASSERT_EQUAL_SHAPE(SymInt[] input1, SymInt[] input2, *, bool allow_broadcast) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def ASSERT_EQUAL_SHAPE(
    input1: list[IntLikeType],
    input2: list[IntLikeType],
    *,
    allow_broadcast: bool,
) -> None:
    _require_shape_extension("ASSERT_EQUAL_SHAPE")
    if len(input1) != len(input2):
        raise TosaValueError(
            "ASSERT_EQUAL_SHAPE requires equal lengths, got "
            f"{len(input1)} and {len(input2)}",
            op="ASSERT_EQUAL_SHAPE",
        )


@register_fake_tosa_op(
    "CONCAT_SHAPE(SymInt[][] shape_list) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def CONCAT_SHAPE(shape_list: list[list[IntLikeType]]) -> list[IntLikeType]:
    _require_shape_extension("CONCAT_SHAPE")
    if not shape_list:
        raise TosaValueError(
            "CONCAT_SHAPE requires at least one shape tensor",
            op="CONCAT_SHAPE",
        )
    if any(not shape for shape in shape_list):
        raise TosaValueError(
            "CONCAT_SHAPE disallows empty input shapes",
            op="CONCAT_SHAPE",
        )

    concat_shape: list[IntLikeType] = []
    for shape in shape_list:
        concat_shape.extend(shape)

    return concat_shape


@register_fake_tosa_op(
    "CONST_SHAPE(int[] shape) -> int[]",
    TosaSpecification.all_versions_and_profiles(),
)
def CONST_SHAPE(shape: list[int]) -> list[int]:
    return shape


@register_fake_tosa_op(
    "DIM(Tensor input, *, int axis) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def DIM(x: torch.Tensor, *, axis: int) -> list[IntLikeType]:
    _require_shape_extension("DIM")
    tosa_spec = get_context_spec()
    supported_dtypes = _supported_dim_dtypes(tosa_spec)
    if x.dtype not in supported_dtypes:
        raise TosaValueError(
            f"Unsupported dtype {x.dtype} for DIM. Supported dtypes are {supported_dtypes}",
            op="DIM",
        )
    if axis < 0 or axis >= x.dim():
        raise TosaValueError(
            f"DIM axis {axis} is out of range for rank {x.dim()}",
            op="DIM",
        )
    _require_known_positive(x.shape[axis], "DIM", "shape[axis]")
    return [x.shape[axis]]


@register_fake_tosa_op(
    "DIV_CEIL_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def DIV_CEIL_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("DIV_CEIL_SHAPE")
    for lhs, rhs in zip(shape1, shape2):
        _require_known_nonnegative(lhs, "DIV_CEIL_SHAPE", "input1")
        _require_known_positive(rhs, "DIV_CEIL_SHAPE", "input2")
    return _combine_shapes(
        shape1,
        shape2,
        lambda a, b: FloorDiv(a + b - sympy.Integer(1), b),
    )


@register_fake_tosa_op(
    "DIV_FLOOR_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def DIV_FLOOR_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("DIV_FLOOR_SHAPE")
    for lhs, rhs in zip(shape1, shape2):
        _require_known_nonnegative(lhs, "DIV_FLOOR_SHAPE", "input1")
        _require_known_positive(rhs, "DIV_FLOOR_SHAPE", "input2")
    return _combine_shapes(shape1, shape2, lambda a, b: FloorDiv(a, b))


@register_fake_tosa_op(
    "EXP2_SHAPE(SymInt[] input) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def EXP2_SHAPE(input: list[IntLikeType]) -> list[IntLikeType]:
    _require_shape_extension("EXP2_SHAPE")
    max_log2_size = 31 if getattr(get_context_spec(), "level_8k", False) else 63
    for value in input:
        _require_known_nonnegative(value, "EXP2_SHAPE", "input")
        _require_known_less_than(value, max_log2_size, "EXP2_SHAPE", "input")
    return _combine_shapes(
        input,
        [2] * len(input),
        lambda a, _: sympy.Integer(2) ** a,
    )


@register_fake_tosa_op(
    "LOG2_CEIL_SHAPE(SymInt[] input) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def LOG2_CEIL_SHAPE(input: list[IntLikeType]) -> list[IntLikeType]:
    _require_shape_extension("LOG2_CEIL_SHAPE")
    for value in input:
        _require_known_positive(value, "LOG2_CEIL_SHAPE", "input")
    return _combine_shapes(
        input,
        [0] * len(input),
        lambda a, _: sympy.ceiling(sympy.log(a, 2)),
    )


@register_fake_tosa_op(
    "LOG2_FLOOR_SHAPE(SymInt[] input) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def LOG2_FLOOR_SHAPE(input: list[IntLikeType]) -> list[IntLikeType]:
    _require_shape_extension("LOG2_FLOOR_SHAPE")
    for value in input:
        _require_known_positive(value, "LOG2_FLOOR_SHAPE", "input")
    return _combine_shapes(
        input,
        [0] * len(input),
        lambda a, _: sympy.floor(sympy.log(a, 2)),
    )


@register_fake_tosa_op(
    "MAX_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MAX_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("MAX_SHAPE")
    return _combine_shapes(shape1, shape2, lambda a, b: sympy.Max(a, b))


@register_fake_tosa_op(
    "MIN_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MIN_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("MIN_SHAPE")
    return _combine_shapes(shape1, shape2, lambda a, b: sympy.Min(a, b))


@register_fake_tosa_op(
    "MOD_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MOD_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("MOD_SHAPE")
    for lhs, rhs in zip(shape1, shape2):
        _require_known_nonnegative(lhs, "MOD_SHAPE", "input1")
        _require_known_positive(rhs, "MOD_SHAPE", "input2")
    return _combine_shapes(shape1, shape2, lambda a, b: a % b)


@register_fake_tosa_op(
    "MUL_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MUL_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("MUL_SHAPE")
    return _combine_shapes(shape1, shape2, lambda a, b: a * b)


@register_fake_tosa_op(
    "SLICE_SHAPE(SymInt[] input, SymInt[] start, SymInt[] size) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def SLICE_SHAPE(
    input: list[IntLikeType],
    start: list[IntLikeType],
    size: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("SLICE_SHAPE")
    if len(start) != 1 or len(size) != 1:
        raise TosaValueError(
            "SLICE_SHAPE requires start[1] and size[1]",
            op="SLICE_SHAPE",
        )

    size_value = _to_lowest_concrete_int(size[0], "SLICE_SHAPE", "size")
    if size_value <= 0:
        raise TosaValueError("SLICE_SHAPE requires size > 0", op="SLICE_SHAPE")

    start_values = _to_finite_int_values(
        start[0],
        "SLICE_SHAPE",
        "start",
        max_values=len(input),
    )
    if start_values is None:
        raise TosaValueError(
            "SLICE_SHAPE requires compile-time constant start or a bounded symbolic "
            "start with finitely many valid values",
            op="SLICE_SHAPE",
        )
    if any(start_value < 0 for start_value in start_values):
        raise TosaValueError("SLICE_SHAPE requires start >= 0", op="SLICE_SHAPE")
    if any(start_value + size_value > len(input) for start_value in start_values):
        raise TosaValueError(
            "SLICE_SHAPE requires start + size within input bounds",
            op="SLICE_SHAPE",
        )

    if len(start_values) == 1:
        start_value = start_values[0]
        return list(input[start_value : start_value + size_value])

    start_expr = _to_sympy_expr(start[0])
    result: list[IntLikeType] = []
    for offset in range(size_value):
        expr = sympy.Piecewise(
            *[
                (
                    _to_sympy_expr(input[start_value + offset]),
                    sympy.Eq(start_expr, sympy.Integer(start_value)),
                )
                for start_value in start_values
            ]
        )
        if expr.is_number and expr.is_integer:
            result.append(int(expr))
            continue

        shape_env = get_context_shape_env()
        result.append(shape_env.create_symintnode(expr, hint=None))
    return result


@register_fake_tosa_op(
    "SUB_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",
    TosaSpecification.all_profiles_for_version("1.1"),
)
def SUB_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    _require_shape_extension("SUB_SHAPE")
    return _combine_shapes(shape1, shape2, lambda a, b: a - b)
