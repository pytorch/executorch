# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import sympy  # type: ignore[import-untyped]
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.interp import sympy_interp

_MAX_SET_SIZE = 256
_ExactValues = Optional[frozenset[sympy.Basic]]


def _expr_to_int(sym_expr: sympy.Basic) -> Optional[int]:
    if isinstance(sym_expr, int):
        return sym_expr
    if isinstance(sym_expr, sympy.Integer):
        return int(sym_expr)
    if getattr(sym_expr, "is_integer", False) and sym_expr.is_number:
        return int(sym_expr)
    return None


def _symbol_values(symbol: sympy.Symbol, shape_env: ShapeEnv) -> _ExactValues:
    value_range = shape_env.var_to_range.get(symbol)
    if value_range is None or not value_range.is_int:
        return None

    lower = _expr_to_int(value_range.lower)
    upper = _expr_to_int(value_range.upper)
    if lower is None or upper is None or upper < lower:
        return None
    if upper - lower + 1 > _MAX_SET_SIZE:
        return None

    return frozenset(sympy.Integer(value) for value in range(lower, upper + 1))


def _map_values(values: _ExactValues, fn) -> _ExactValues:
    if values is None:
        return None

    result = {sympy.simplify(fn(value)) for value in values}
    if len(result) > _MAX_SET_SIZE:
        return None
    return frozenset(result)


def _combine_values(lhs: _ExactValues, rhs: _ExactValues, fn) -> _ExactValues:
    if lhs is None or rhs is None:
        return None
    if len(lhs) * len(rhs) > _MAX_SET_SIZE * _MAX_SET_SIZE:
        return None

    result = {sympy.simplify(fn(a, b)) for a in lhs for b in rhs}
    if len(result) > _MAX_SET_SIZE:
        return None
    return frozenset(result)


class _ExactValueAnalysis:
    @staticmethod
    def constant(value, dtype) -> frozenset[sympy.Basic]:
        return frozenset({sympy.sympify(value)})

    @staticmethod
    def add(lhs: _ExactValues, rhs: _ExactValues) -> _ExactValues:
        return _combine_values(lhs, rhs, lambda a, b: a + b)

    @staticmethod
    def mul(lhs: _ExactValues, rhs: _ExactValues) -> _ExactValues:
        return _combine_values(lhs, rhs, lambda a, b: a * b)

    @staticmethod
    def mod(lhs: _ExactValues, rhs: _ExactValues) -> _ExactValues:
        if rhs is None or any(value == 0 for value in rhs):
            return None
        return _combine_values(lhs, rhs, lambda a, b: sympy.Mod(a, b))

    @staticmethod
    def pow(lhs: _ExactValues, rhs: _ExactValues) -> _ExactValues:
        return _combine_values(lhs, rhs, lambda a, b: a**b)

    @staticmethod
    def floor_to_int(values: _ExactValues, dtype) -> _ExactValues:
        return _map_values(values, sympy.floor)

    @staticmethod
    def sym_sum(args: list[_ExactValues]) -> _ExactValues:
        acc: _ExactValues = frozenset({sympy.Integer(0)})
        for arg in args:
            acc = _ExactValueAnalysis.add(acc, arg)
            if acc is None:
                return None
        return acc


def evaluate_symbolic_expr_values(
    expr: sympy.Basic | torch.SymInt,
    shape_env: ShapeEnv,
) -> Optional[set[int]]:
    """Return a best-effort finite set of possible integer values.

    The helper first relies on ``bound_sympy`` for cheap singleton detection.
    When interval bounds are not precise enough, it falls back to a small
    exact-set analysis over bounded symbols using ``sympy_interp``.

    """
    root_expr = sympy.simplify(
        expr.node.expr if isinstance(expr, torch.SymInt) else expr
    )
    value_range = shape_env.bound_sympy(root_expr)
    if value_range.is_int and value_range.is_singleton():
        singleton = _expr_to_int(value_range.lower)
        return {singleton} if singleton is not None else None

    exact_values = sympy_interp(
        _ExactValueAnalysis,
        {
            symbol: _symbol_values(symbol, shape_env)
            for symbol in root_expr.free_symbols
        },
        root_expr,
        missing_handler=lambda symbol: _symbol_values(symbol, shape_env),
    )
    if exact_values is None:
        return None

    result: set[int] = set()
    for value in exact_values:
        integer_value = _expr_to_int(sympy.simplify(value))
        if integer_value is None:
            return None
        result.add(integer_value)
    return result
