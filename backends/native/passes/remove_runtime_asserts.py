# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sympy
import torch

from executorch.exir.pass_base import ExportPass, PassResult
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges


class RemoveRuntimeAssertsPass(ExportPass):
    """Remove export-time guards that PTN input bounds already enforce.

    A PTN records each input dim as a [min, max] range and an engine rejects sizes
    outside it, so a guard that holds for every size in those ranges is redundant,
    as is a tensor metadata assert. Any other guard (divisibility, data-dependent
    values) is kept: it has no native kernel, so lowering fails instead of
    dropping the check.
    """

    _SYM_RANGE_OPS = {
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        ranges = _input_dim_ranges(graph_module)
        changed = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            erased = False
            for node in list(module.graph.nodes):
                if node.op == "call_function" and self._redundant(node, ranges):
                    module.graph.erase_node(node)
                    erased = True
            if erased:
                module.graph.eliminate_dead_code()
                module.recompile()
                changed = True
        return PassResult(graph_module, changed)

    def _redundant(
        self, node: torch.fx.Node, ranges: dict[sympy.Symbol, ValueRanges]
    ) -> bool:
        if node.target is torch.ops.aten._assert_tensor_metadata.default:
            # PTN tensor values carry their dtype.
            return True
        if node.target is torch.ops.aten._assert_scalar.default:
            cond = node.args[0]
            if isinstance(cond, bool):
                return cond
            val = cond.meta.get("val") if isinstance(cond, torch.fx.Node) else None
            return isinstance(val, torch.SymBool) and _always_true(
                val.node.expr, ranges
            )
        if node.target in self._SYM_RANGE_OPS:
            size = node.args[0]
            val = size.meta.get("val") if isinstance(size, torch.fx.Node) else size
            if not isinstance(val, (int, torch.SymInt)):
                return False
            expr = sympy.sympify(
                val.node.expr if isinstance(val, torch.SymInt) else val
            )
            lo = node.kwargs.get("min")
            hi = node.kwargs.get("max")
            return _always_true(
                sympy.And(
                    expr >= (lo if lo is not None else -sympy.oo),
                    expr <= (hi if hi is not None else sympy.oo),
                ),
                ranges,
            )
        return False


def _input_dim_ranges(
    graph_module: torch.fx.GraphModule,
) -> dict[sympy.Symbol, ValueRanges]:
    """The range PTN enforces for each symbol that is an input dim."""
    ranges: dict[sympy.Symbol, ValueRanges] = {}
    for node in graph_module.graph.nodes:
        val = node.meta.get("val") if node.op == "placeholder" else None
        if not isinstance(val, torch.Tensor):
            continue
        for dim in val.shape:
            if not isinstance(dim, torch.SymInt):
                continue
            expr = dim.node.expr
            if not isinstance(expr, sympy.Symbol):
                continue
            vr = dim.node.shape_env.var_to_range[expr]
            # The ShapeEnv raises a declared min of 0 or 1 to 2, and the declared
            # value is not visible here, so assume the widest.
            lower = 0 if vr.lower == 2 else vr.lower
            ranges[expr] = ValueRanges(lower, vr.upper)
    return ranges


def _always_true(expr: sympy.Basic, ranges: dict[sympy.Symbol, ValueRanges]) -> bool:
    if expr is sympy.true:
        return True
    if not expr.free_symbols <= ranges.keys():
        return False
    try:
        vr = bound_sympy(expr, ranges)
    except (TypeError, ValueError):
        return False
    return vr.is_singleton() and vr.lower is sympy.true
