# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import sympy  # type: ignore[import-untyped]
import torch


def materialize_symints(
    graph: torch.fx.Graph, values: Sequence[int | torch.SymInt]
) -> list[torch.fx.Node | int]:
    """Materialize symbolic integers into FX graph values."""
    materialized = graph.materialize_symints(values)
    if any(isinstance(value, torch.SymInt) for value in materialized):
        raise AssertionError("materialize_symints returned a raw SymInt")
    return materialized


def _shape_source_symbol(expr: sympy.Expr) -> sympy.Symbol | None:
    coefficient, symbol = expr.as_coeff_Mul()
    if (
        isinstance(coefficient, sympy.Integer)
        and coefficient > 0
        and isinstance(symbol, sympy.Symbol)
    ):
        return symbol
    return None


def can_materialize_symints(
    graph: torch.fx.Graph,
    values: Sequence[int | torch.SymInt],
    before: torch.fx.Node,
) -> bool:
    """Check that every symbol has a source before changing the graph."""
    if not any(isinstance(value, torch.SymInt) for value in values):
        return True

    available: set[sympy.Symbol] = set()
    for node in graph.nodes:
        if node is before:
            break
        if node.op == "placeholder":
            value = node.meta.get("val", node.meta.get("example_value"))
            if isinstance(value, torch.SymInt):
                expr = value.node.expr
                if isinstance(expr, sympy.Symbol):
                    available.add(expr)
            elif isinstance(value, torch.Tensor):
                for dim in value.shape:
                    if isinstance(dim, torch.SymInt):
                        symbol = _shape_source_symbol(dim.node.expr)
                        if symbol is not None:
                            available.add(symbol)
        else:
            available.update(node.meta.get("unbacked_bindings", ()))

    return all(
        not isinstance(value, torch.SymInt) or value.node.expr.free_symbols <= available
        for value in values
    )
