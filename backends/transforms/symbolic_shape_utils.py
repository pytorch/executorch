# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, NamedTuple, Sequence

import sympy  # type: ignore[import-untyped]
import torch
from sympy.logic.boolalg import BooleanAtom  # type: ignore[import-untyped]
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import PythonReferenceAnalysis


class _SymSizeSource(NamedTuple):
    placeholder: torch.fx.Node
    dim: int
    dim_value: torch.SymInt
    symbol: sympy.Symbol
    coefficient: int


def _expr_single_symbol_multiple(expr: sympy.Expr) -> tuple[sympy.Symbol, int] | None:
    if isinstance(expr, sympy.Symbol):
        return expr, 1

    coefficient, factors = expr.as_coeff_mul()
    if (
        len(factors) == 1
        and isinstance(factors[0], sympy.Symbol)
        and coefficient.is_integer
    ):
        return factors[0], int(coefficient)
    return None


def _node_value(node: torch.fx.Node) -> Any:
    value = node.meta.get("val")
    if value is not None:
        return value
    return node.meta.get("example_value")


def _set_node_value(node: torch.fx.Node, value: Any) -> None:
    node.meta["val"] = value
    # val is canonical. Mirror it to example_value when inputs use
    # that metadata convention.
    if any(
        isinstance(arg, torch.fx.Node) and "example_value" in arg.meta
        for arg in node.args
    ):
        node.meta["example_value"] = value


def _existing_sym_size_node_before(
    graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
    dim: int,
    before_node: torch.fx.Node,
) -> torch.fx.Node | None:
    for node in graph.nodes:
        if node is before_node:
            break
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.sym_size.int
            and node.args == (placeholder, dim)
        ):
            return node
    return None


def _needed_symbols(values: Sequence[int | torch.SymInt]) -> set[sympy.Symbol]:
    return {
        symbol
        for value in values
        if isinstance(value, torch.SymInt)
        for symbol in value.node.expr.free_symbols
    }


def _collect_sym_size_sources(
    graph: torch.fx.Graph,
    needed_symbols: set[sympy.Symbol],
    tracer: torch.fx.proxy.GraphAppendingTracer,
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy],
) -> list[_SymSizeSource]:
    sources: list[_SymSizeSource] = []
    for node in graph.nodes:
        if node.op != "placeholder":
            continue

        meta_value = _node_value(node)
        if isinstance(meta_value, py_sym_types):
            expr = meta_value.node.expr
            if isinstance(expr, sympy.Symbol) and expr in needed_symbols:
                expr_to_proxy.setdefault(expr, torch.fx.Proxy(node, tracer=tracer))
            continue

        if isinstance(meta_value, torch.Tensor):
            sources.extend(_sources_from_tensor(node, meta_value, needed_symbols))
    return sources


def _sources_from_tensor(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    needed_symbols: set[sympy.Symbol],
) -> list[_SymSizeSource]:
    sources: list[_SymSizeSource] = []
    for dim, dim_value in enumerate(tensor.shape):
        if not isinstance(dim_value, torch.SymInt):
            continue
        source = _expr_single_symbol_multiple(dim_value.node.expr)
        if source is None:
            continue
        symbol, coefficient = source
        if symbol in needed_symbols:
            sources.append(_SymSizeSource(node, dim, dim_value, symbol, coefficient))
    return sources


def _add_sym_size_source_proxies(
    graph: torch.fx.Graph,
    tracer: torch.fx.proxy.GraphAppendingTracer,
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy],
    sources: list[_SymSizeSource],
) -> None:
    for source in sources:
        if source.symbol in expr_to_proxy:
            continue

        inserted_sym_size_node = graph.call_function(
            torch.ops.aten.sym_size.int, (source.placeholder, source.dim)
        )
        sym_size_node = _existing_sym_size_node_before(
            graph,
            source.placeholder,
            source.dim,
            inserted_sym_size_node,
        )
        if sym_size_node is None:
            sym_size_node = inserted_sym_size_node
            _set_node_value(sym_size_node, source.dim_value)
        else:
            graph.erase_node(inserted_sym_size_node)

        sym_size_proxy = torch.fx.Proxy(sym_size_node, tracer=tracer)
        expr_to_proxy.setdefault(source.dim_value.node.expr, sym_size_proxy)
        if source.coefficient == 1:
            expr_to_proxy.setdefault(source.symbol, sym_size_proxy)
            continue

        symbol_node = graph.call_function(
            operator.floordiv, (sym_size_node, source.coefficient)
        )
        _set_node_value(symbol_node, source.dim_value // source.coefficient)
        expr_to_proxy.setdefault(
            source.symbol, torch.fx.Proxy(symbol_node, tracer=tracer)
        )


def _arg_meta_value(arg: Any) -> Any:
    if isinstance(arg, torch.fx.Node):
        return _node_value(arg)
    return arg


def _annotate_proxy_meta(result: torch.fx.Proxy) -> None:
    target = result.node.target
    if not callable(target):
        return
    try:
        fake_args = tuple(_arg_meta_value(arg) for arg in result.node.args)
        fake_kwargs = {
            key: _arg_meta_value(value) for key, value in result.node.kwargs.items()
        }
        _set_node_value(result.node, target(*fake_args, **fake_kwargs))
    except (TypeError, ValueError, AttributeError):
        pass


def _sympy_interp_cached(
    expr: sympy.Expr,
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy],
) -> Any:
    if expr in expr_to_proxy:
        return expr_to_proxy[expr]
    if isinstance(expr, (sympy.Integer, sympy.Number, sympy.Symbol, BooleanAtom)):
        return sympy_interp(PythonReferenceAnalysis, expr_to_proxy, expr)

    result = _run_sympy_handler(
        PythonReferenceAnalysis,
        [_sympy_interp_cached(arg, expr_to_proxy) for arg in expr.args],
        expr,
    )
    if isinstance(result, torch.fx.Proxy):
        expr_to_proxy[expr] = result
        _annotate_proxy_meta(result)
    return result


def _materialize_values(
    values: Sequence[int | torch.SymInt],
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy],
    original_error: AssertionError,
) -> list[torch.fx.Node | int]:
    out: list[torch.fx.Node | int] = []
    for value in values:
        if isinstance(value, int):
            out.append(value)
            continue
        if not isinstance(value, torch.SymInt):
            raise TypeError(
                f"materialize_symints: expected SymInt or int, got "
                f"{type(value).__name__}"
            )
        out.append(_materialize_value(value, expr_to_proxy, original_error))
    return out


def _materialize_value(
    value: torch.SymInt,
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy],
    original_error: AssertionError,
) -> torch.fx.Node | int:
    target_expr = value.node.expr
    if target_expr.is_number:
        return int(value)
    missing_symbols = [
        symbol for symbol in target_expr.free_symbols if symbol not in expr_to_proxy
    ]
    if missing_symbols:
        raise original_error

    result = _sympy_interp_cached(target_expr, expr_to_proxy)
    if not isinstance(result, torch.fx.Proxy):
        if not isinstance(result, (int, bool)):
            raise AssertionError(
                f"materialize_symints: non-Proxy result {result!r} for "
                f"non-constant target {target_expr!r}"
            )
        return int(result)
    _set_node_value(result.node, value)
    return result.node


def _materialize_from_placeholder_dims(
    graph: torch.fx.Graph,
    values: Sequence[int | torch.SymInt],
    original_error: AssertionError,
) -> list[torch.fx.Node | int]:
    needed_symbols = _needed_symbols(values)
    tracer = torch.fx.proxy.GraphAppendingTracer(graph)
    expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy] = {}
    sources = _collect_sym_size_sources(graph, needed_symbols, tracer, expr_to_proxy)
    _add_sym_size_source_proxies(graph, tracer, expr_to_proxy, sources)
    return _materialize_values(values, expr_to_proxy, original_error)


def materialize_symints(
    graph: torch.fx.Graph, values: Sequence[int | torch.SymInt]
) -> list[torch.fx.Node | int]:
    """Materialize symbolic integers into FX graph values.

    Falls back to deriving base symbols from scaled symbolic placeholder
    dimensions, for example deriving ``s0`` from an input dim ``16*s0``.

    Args:
        graph (torch.fx.Graph): Graph where producer nodes are inserted.
        values (Sequence[int | torch.SymInt]): Integer values to materialize.

    Returns:
        list[torch.fx.Node | int]: Materialized graph nodes or static integers.

    """
    try:
        materialized = graph.materialize_symints(values)
        if not any(isinstance(value, torch.SymInt) for value in materialized):
            return materialized
        raise AssertionError("materialize_symints returned a raw SymInt")
    except AssertionError as error:
        if "materialize_symints" not in str(error):
            raise

        return _materialize_from_placeholder_dims(graph, values, error)


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
