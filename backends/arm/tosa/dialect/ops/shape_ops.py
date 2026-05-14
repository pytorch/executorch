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


@register_fake_tosa_op(
    "CONST_SHAPE(int[] shape) -> int[]",  # schema
    TosaSpecification.all_versions_and_profiles(),
)
def CONST_SHAPE(shape: list[int]) -> list[int]:
    """CONST_SHAPE operator creates a constant shape tensor."""

    return shape


@register_fake_tosa_op(
    "DIM(Tensor input, *, int axis) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def DIM(x: torch.Tensor, *, axis: int) -> list[torch.SymInt]:
    tosa_spec = get_context_spec()
    """Dim operator extracts a dimension from the input tensor shape."""

    if not tosa_spec.support_extension("shape"):
        raise TosaValueError(
            f"TOSA spec {tosa_spec} doesn't support shape extension", op="DIM"
        )

    assert isinstance(
        x.shape[axis], torch.SymInt
    ), f"Expected dimension to be SymInt, got {type(x.shape[axis])}"
    return [x.shape[axis]]  # type: ignore[list-item]


def _to_sympy_expr(value: IntLikeType) -> sympy.Expr:
    """Lift a shape value to a SymPy expression without forcing hints."""

    if isinstance(value, torch.SymInt):
        # `node.expr` flows through ShapeEnv.replace and would plug in hints.
        # `_expr` is the raw symbolic expression we need to preserve.
        return value.node._expr
    return sympy.Integer(int(value))


def _combine_shapes(
    lhs: list[IntLikeType],
    rhs: list[IntLikeType],
    combine: Callable[[sympy.Expr, sympy.Expr], sympy.Expr | sympy.Integer],
) -> list[IntLikeType]:
    """The fake kernels run during export/meta execution.

    Using Python arithmetic
    directly on `torch.SymInt` would consult the current ShapeEnv hints and
    collapse dynamic symbols to concrete ints.  Instead we work with the
    underlying SymPy expressions and wrap them back into SymInts via the same
    ShapeEnv, preserving dynamic information for later passes.

    """
    assert len(lhs) == len(
        rhs
    ), f"Expected shapes to be of same length, got {len(lhs)} and {len(rhs)}"

    expr_lhs = [_to_sympy_expr(v) for v in lhs]
    expr_rhs = [_to_sympy_expr(v) for v in rhs]

    shape_env = get_context_shape_env()
    result: list[IntLikeType] = []
    for a, b in zip(expr_lhs, expr_rhs):
        expr = combine(a, b)
        if isinstance(expr, sympy.Expr):
            result.append(shape_env.create_symintnode(expr, hint=None))
        else:
            result.append(int(expr))
    return result


@register_fake_tosa_op(
    "CONCAT_SHAPE(SymInt[][] shape_list) -> SymInt[]",  # schema (fixed to return SymInt[])
    TosaSpecification.all_profiles_for_version("1.1"),
)
def CONCAT_SHAPE(
    shape_list: list[list[IntLikeType]],
) -> list[IntLikeType]:
    """CONCAT_SHAPE operator concatenates a list of shape lists to create a new
    list with length the sum of lengths of all lists in input shape_list.
    """

    if len(shape_list) < 1:
        raise TosaValueError(
            f"CONCAT_SHAPE expected 2 or more shape tensors, got {len(shape_list)}",
            op="CONCAT_SHAPE",
        )

    concat_shape = list(shape_list[0])
    for d in shape_list[1:]:
        concat_shape.extend(d)

    return concat_shape


@register_fake_tosa_op(
    "ADD_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def ADD_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    """ADD_SHAPE operator adds each element of the second shape tensor to the
    first.
    """
    return _combine_shapes(shape1, shape2, lambda a, b: a + b)


@register_fake_tosa_op(
    "SUB_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def SUB_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    """SUB_SHAPE operator subtracts each element of the second shape tensor from
    the first.
    """

    return _combine_shapes(shape1, shape2, lambda a, b: a - b)


@register_fake_tosa_op(
    "DIV_FLOOR_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def DIV_FLOOR_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    """DIV_SHAPE operator divides each element of the shape tensor by the given
    denominator.
    """
    return _combine_shapes(shape1, shape2, lambda a, b: FloorDiv(a, b))


@register_fake_tosa_op(
    "MUL_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MUL_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    """MUL_SHAPE operator multiplies each element of the shape tensor by the
    given factor.
    """

    return _combine_shapes(shape1, shape2, lambda a, b: a * b)


@register_fake_tosa_op(
    "MOD_SHAPE(SymInt[] shape1, SymInt[] shape2) -> SymInt[]",  # schema
    TosaSpecification.all_profiles_for_version("1.1"),
)
def MOD_SHAPE(
    shape1: list[IntLikeType],
    shape2: list[IntLikeType],
) -> list[IntLikeType]:
    """MOD_SHAPE operator computes the element-wise modulo of the first shape
    tensor by the second.
    """

    return _combine_shapes(shape1, shape2, lambda a, b: a % b)
