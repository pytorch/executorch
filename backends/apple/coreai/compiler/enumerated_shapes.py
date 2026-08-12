# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Enumerated input-shape propagation for the Core AI backend.

Enumerations are declared on the *ET model inputs*, resolved to ``torch.export``
symbols (partitioner side, :func:`resolve_input_enumerations`), then substituted
into each delegate subgraph's boundary symbolic shapes and attached via
``AIProgram.set_static_shape_config`` (preprocess side,
:func:`apply_enumerated_shapes`).
"""

from itertools import product
from typing import Dict, List, Optional, Sequence

from executorch.backends.apple.coreai.compiler.constants import MAIN_ENTRYPOINT
from torch.export.exported_program import ExportedProgram


def resolve_input_enumerations(
    exported_program: ExportedProgram,
    input_enumerations: Optional[Sequence[Optional[Dict[int, Sequence[int]]]]],
) -> Optional[Dict[str, List[int]]]:
    """Map ET-input ``(index, dim)`` enumerations to ``{symbol_name: [values]}``.

    Reads the export symbol backing each enumerated ET-input dim from the
    exported program; raises if the dim is static (no symbol) or maps to a
    non-single-symbol expression.
    """
    if not input_enumerations:
        return None
    placeholders = {
        n.name: n for n in exported_program.graph.nodes if n.op == "placeholder"
    }
    user_inputs = list(exported_program.graph_signature.user_inputs)
    resolved: Dict[str, List[int]] = {}
    for i, dim_map in enumerate(input_enumerations):
        if not dim_map:
            continue
        if i >= len(user_inputs):
            raise ValueError(
                f"input_enumerations has an entry for input {i} but the model "
                f"has {len(user_inputs)} input(s)"
            )
        val = (
            placeholders[user_inputs[i]].meta.get("val")
            if (isinstance(user_inputs[i], str) and user_inputs[i] in placeholders)
            else None
        )
        if not hasattr(val, "shape"):
            raise ValueError(
                f"input {i} is not a tensor input; cannot enumerate its shape"
            )
        for dim_index, values in dim_map.items():
            if not 0 <= dim_index < len(val.shape):
                raise ValueError(
                    f"input {i} has {len(val.shape)} dim(s); cannot enumerate "
                    f"dim {dim_index}"
                )
            values = list(values)
            if not values:
                raise ValueError(
                    f"input {i} dim {dim_index} has no enumerated values; drop "
                    "the entry instead of passing an empty list"
                )
            dim = val.shape[dim_index]
            if isinstance(dim, int):
                raise ValueError(
                    f"input {i} dim {dim_index} is static ({dim}); export it "
                    "with dynamic_shapes to enumerate it"
                )
            symbols = dim.node.expr.free_symbols
            if len(symbols) != 1:
                raise ValueError(
                    f"input {i} dim {dim_index} shape {dim.node.expr} is not a "
                    "single symbol; cannot enumerate"
                )
            symbol = next(iter(symbols))
            _check_within_range(exported_program, symbol, values, i, dim_index)
            name = str(symbol)
            if name in resolved and resolved[name] != values:
                raise ValueError(
                    f"input {i} dim {dim_index} shares symbol {name} with an "
                    f"earlier input but enumerates different values "
                    f"{values} vs {resolved[name]}; dims tied to one Dim must "
                    "agree"
                )
            resolved[name] = values
    return resolved


def _check_within_range(exported_program, symbol, values, input_index, dim_index):
    """Reject values the exported dim cannot take.

    Out-of-range values are accepted by the shape config and only fail later,
    when the compiler or runtime rejects a specialization it cannot map back to
    any input.
    """
    bad = [v for v in values if v <= 0]
    if bad:
        raise ValueError(
            f"input {input_index} dim {dim_index} enumerates non-positive "
            f"value(s) {bad}"
        )
    constraint = exported_program.range_constraints.get(symbol)
    if constraint is None:
        return
    low, high = constraint.lower, constraint.upper
    outside = [v for v in values if v < low or v > high]
    if outside:
        raise ValueError(
            f"input {input_index} dim {dim_index} enumerates {outside}, "
            f"outside the exported range [{low}, {high}]"
        )


def graph_input_names(program, entrypoint: str = MAIN_ENTRYPOINT) -> List[str]:
    """Names of the converted coreai graph's inputs (in argument order)."""
    graph = program.get_graph(entrypoint)
    names = []
    for attr in graph.arg_attrs:
        if "coreai.name" in attr:
            names.append(attr["coreai.name"].value)
    return names


def _eval_dim(dim, subs: Dict[str, int], uname: str, axis: int) -> int:
    """Resolve a possibly-symbolic shape dim to a concrete int under ``subs``.

    A dim can mix an enumerated symbol with one that was left dynamic (batch
    enumerated, sequence not). Substitution leaves that symbolic, and ``int()``
    would raise ``Cannot convert symbols to int`` naming neither the input nor
    the dim, so report it here instead.
    """
    if isinstance(dim, int):
        return dim
    expr = dim.node.expr
    mapping = {s: subs[str(s)] for s in expr.free_symbols if str(s) in subs}
    resolved = expr.subs(mapping)
    if resolved.free_symbols:
        raise ValueError(
            f"input '{uname}' dim {axis} is {expr}, which still depends on "
            f"{sorted(str(s) for s in resolved.free_symbols)} after "
            f"substitution; enumerate those dims too, or leave this input "
            f"fully dynamic"
        )
    return int(resolved)


def apply_enumerated_shapes(
    program,
    edge_program: ExportedProgram,
    enumerations: Dict[str, Sequence[int]],
) -> None:
    """Attach enumerated shapes to the coreai program (all delivery modes).

    ``enumerations`` is ``{symbol_name: [value, ...]}`` (from
    :func:`resolve_input_enumerations`).  For each of this subgraph's user inputs
    we read its symbolic shape from ``edge_program`` and substitute every
    combination of the enumerated symbol values, then attach the resulting shapes
    via ``set_static_shape_config``.  Inputs are matched to coreai graph inputs by
    name (the converter names each coreai input after its edge placeholder).
    """
    if not enumerations:
        return

    placeholders = {
        n.name: n for n in edge_program.graph.nodes if n.op == "placeholder"
    }
    user_inputs = list(edge_program.graph_signature.user_inputs)
    coreai_names = set(graph_input_names(program))

    # Collect the inputs to specialize first: one entry per combination has to
    # name every one of them, since set_static_shape_config treats a key as a
    # whole-graph shape. Enumerating each input separately would leave the
    # others unconstrained in every entry, and inputs sharing a symbol (a batch
    # dim, say) would not be held equal.
    enumerated_inputs = []
    active_symbols = set()
    for uname in user_inputs:
        # Match edge user inputs to coreai graph inputs by name; skip anything
        # that isn't a coreai tensor input (e.g. a taken-over constant).
        if not isinstance(uname, str) or uname not in coreai_names:
            continue
        node = placeholders.get(uname)
        val = node.meta.get("val") if node is not None else None
        if not hasattr(val, "shape"):
            continue
        dims = list(val.shape)
        symbols = set()
        for d in dims:
            if not isinstance(d, int):
                symbols |= {str(s) for s in d.node.expr.free_symbols}
        enumerated = symbols & set(enumerations)
        if not enumerated:
            continue
        enumerated_inputs.append((uname, dims))
        active_symbols |= enumerated

    if not enumerated_inputs:
        return

    # One axis per enumerated symbol, shared across every input that uses it.
    active = [s for s in enumerations if s in active_symbols]
    shapes_config: Dict[str, Dict[str, tuple]] = {}
    for combo in product(*[enumerations[s] for s in active]):
        subs = dict(zip(active, combo))
        # Name the specialization after the values it pins. The key becomes an
        # MLIR dictionary attribute name, so it has to stay an identifier: no
        # "=" or other punctuation.
        key = "_".join(f"{s}_{v}" for s, v in subs.items())
        shapes_config[key] = {
            uname: tuple(_eval_dim(d, subs, uname, axis) for axis, d in enumerate(dims))
            for uname, dims in enumerated_inputs
        }

    if shapes_config:
        program.set_static_shape_config(MAIN_ENTRYPOINT, shapes_config)
