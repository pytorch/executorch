# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes.symbolic_value_range import (
    evaluate_symbolic_expr_values,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _make_shape_env(
    *,
    symbol_name: str = "s89",
    hint: int = 2,
    compiler_min: int = 1,
    compiler_max: int = 2,
) -> tuple[ShapeEnv, torch.SymInt]:
    shape_env = ShapeEnv()
    symint = shape_env.create_symintnode(sympy.Symbol(symbol_name), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr,
        compiler_min=compiler_min,
        compiler_max=compiler_max,
    )
    return shape_env, symint


def test_evaluate_symbolic_expr_values_returns_singleton_for_constant_expr() -> None:
    shape_env, symint = _make_shape_env()

    assert evaluate_symbolic_expr_values(
        symint.node.expr - symint.node.expr, shape_env
    ) == {0}
    assert evaluate_symbolic_expr_values(
        sympy.floor(symint.node.expr / symint.node.expr), shape_env
    ) == {1}


def test_evaluate_symbolic_expr_values_returns_singleton_for_singleton_symint() -> None:
    shape_env, symint = _make_shape_env(hint=3, compiler_min=3, compiler_max=3)

    assert evaluate_symbolic_expr_values(symint, shape_env) == {3}
    assert evaluate_symbolic_expr_values(symint.node.expr, shape_env) == {3}


def test_evaluate_symbolic_expr_values_enumerates_non_singleton_symint() -> None:
    shape_env, symint = _make_shape_env(hint=3, compiler_min=2, compiler_max=6)

    assert evaluate_symbolic_expr_values(symint, shape_env) == {2, 3, 4, 5, 6}
    assert evaluate_symbolic_expr_values(symint.node.expr, shape_env) == {2, 3, 4, 5, 6}


def test_evaluate_symbolic_expr_values_tracks_exact_modulo_residue() -> None:
    shape_env, symint = _make_shape_env(hint=3, compiler_min=2, compiler_max=6)
    expr = sympy.Mod(16 * symint.node.expr - 7, 4)

    value_range = shape_env.bound_sympy(expr)
    assert value_range.lower == 0
    assert value_range.upper == 3
    assert evaluate_symbolic_expr_values(expr, shape_env) == {1}


def test_evaluate_symbolic_expr_values_bails_out_for_large_symbol_ranges() -> None:
    shape_env, symint = _make_shape_env(hint=3, compiler_min=1, compiler_max=400)

    assert evaluate_symbolic_expr_values(symint, shape_env) is None
