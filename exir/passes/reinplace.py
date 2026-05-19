# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Tuple

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.operator.convert import (
    output_to_aliased_input_map,
    unwrap_op_overload,
)
from torch.export import ExportedProgram


# ---------------------------------------------------------------------------
# Public API for extending the pass with additional ops.
# ---------------------------------------------------------------------------


# Default set of edge-dialect functional ops the pass attempts to rewrite
# in-place. Today only `index_put -> index_put_`. The pass auto-derives
# the in-place form by name + schema match — callers who add ops here
# do not need to specify the in-place op explicitly.
#
# `reinplace_pass` runs after `to_edge` (inside `to_executorch`), so it
# only ever sees edge-dialect targets (`EdgeOpOverload`). Aten targets
# do not appear at this stage; only edge ops belong here.
DEFAULT_INPLACEABLE_OPS: FrozenSet[Any] = frozenset(
    {
        ops.edge.aten.index_put.default,
    }
)


# ---------------------------------------------------------------------------
# Schema-based discovery and validation.
# ---------------------------------------------------------------------------


def _op_schema(op: Any) -> torch.FunctionSchema:
    """Return the underlying ``FunctionSchema`` for an op overload.

    Delegates to ``unwrap_op_overload`` to peel any edge-dialect or
    backend wrapper down to the bare ``torch._ops.OpOverload`` before
    reading ``_schema``. Falls back to ``op._schema`` if the input is
    a schema-bearing object that ``unwrap_op_overload`` doesn't
    recognize (e.g., a custom op-like with a ``_schema`` attribute but
    not a true ``OpOverload``).
    """
    try:
        return unwrap_op_overload(op)._schema
    except TypeError:
        return op._schema


def _is_inplace_of(
    functional_schema: torch.FunctionSchema,
    candidate_schema: torch.FunctionSchema,
) -> bool:
    """Return True if `candidate_schema` is the in-place form of
    `functional_schema`: same input arg types positionally, with the
    first arg of the candidate carrying a `Tensor(a!)` write alias.
    """
    f_args = functional_schema.arguments
    c_args = candidate_schema.arguments
    if len(f_args) != len(c_args):
        return False
    if not c_args:
        return False
    first = c_args[0]
    if first.alias_info is None or not first.alias_info.is_write:
        return False
    for fa, ca in zip(f_args, c_args):
        # Compare JIT types directly; `str(...)` repr equality is
        # fragile to type-printing changes (qualified vs. unqualified
        # names, container parameterization). Fall back to str only if
        # direct equality raises (some C-bound types may not implement
        # __eq__ uniformly across Torch versions).
        try:
            if fa.type != ca.type:
                return False
        except Exception:
            if str(fa.type) != str(ca.type):
                return False
    return True


def _derive_edge_inplace_overload(functional_op: Any) -> Optional[Any]:
    """Auto-derive the in-place edge-dialect overload for a functional
    op (works for both `EdgeOpOverload` inputs — the common case —
    and bare aten `OpOverload` inputs; the result is always an
    edge-dialect op).

    Strategy: peel `EdgeOpOverload._op` (no-op for bare aten) to find
    the underlying aten schema. The aten in-place form lives at
    `torch.ops.aten.<base>_` (for example,
    `aten::index_put -> aten::index_put_`). Walk the aten in-place
    package's overloads (which exposes `.overloads()` cleanly; the
    edge package does not) to find the one whose schema matches the
    functional schema (with the first arg promoted to `Tensor(a!)`).
    Then look up the corresponding edge-dialect op at
    `ops.edge.aten.<base>_.<overload_name>`.

    Handles cross-overload-name asymmetry — e.g.
        `aten::pow.Tensor_Scalar` -> `aten::pow_.Scalar`
    where simple name-based lookup `aten.pow_.Tensor_Scalar` fails.

    Note: if multiple aten in-place overloads have schemas that match
    the functional op's schema (no known cases in aten today), the
    first one returned by `.overloads()` wins. Callers can disambiguate
    by passing an explicit `inplace_overrides` entry.

    Returns None if no schema match is found, or the op is not an
    aten op. Callers should provide an explicit override via
    `inplace_overrides` for non-conventional rewrites.
    """
    schema = _op_schema(functional_op)
    name = schema.name  # e.g. "aten::index_put"
    if "::" not in name:
        return None
    namespace, base = name.split("::", 1)
    if namespace != "aten":
        return None

    # Find the matching aten in-place overload first (aten exposes
    # `.overloads()` cleanly; the edge package does not).
    aten_inplace_pkg = getattr(torch.ops.aten, base + "_", None)
    if aten_inplace_pkg is None:
        return None
    matched_overload_name: Optional[str] = None
    for overload_name in aten_inplace_pkg.overloads():
        candidate = getattr(aten_inplace_pkg, overload_name, None)
        if candidate is None:
            continue
        cand_schema = getattr(candidate, "_schema", None)
        if cand_schema is None:
            continue
        if _is_inplace_of(schema, cand_schema):
            matched_overload_name = overload_name
            break
    if matched_overload_name is None:
        return None

    # Translate to the edge-dialect op of the same name + overload.
    edge_inplace_pkg = getattr(ops.edge.aten, base + "_", None)
    if edge_inplace_pkg is None:
        return None
    return getattr(edge_inplace_pkg, matched_overload_name, None)


def _validate_inplace_mapping(functional_op: Any, inplace_op: Any) -> None:
    """Validate that `inplace_op` is a plausible in-place form of
    `functional_op`. Raises `ValueError` on misregistration.

    Two checks:
      1. **Schema shape**: same arg types positionally, first arg of
         `inplace_op` carries `Tensor(a!)`. Catches gross mismatches
         like `add -> mul_` (different signatures).
      2. **Name affinity**: in-place op's name starts with the
         functional op's name + "_". Catches subtle mismatches with
         matching schemas like `add -> sub_`.
    """
    f_schema = _op_schema(functional_op)
    i_schema = _op_schema(inplace_op)

    if not _is_inplace_of(f_schema, i_schema):
        raise ValueError(
            f"Schema mismatch in reinplace registration: "
            f"{functional_op} -> {inplace_op}. "
            f"Expected in-place arg types to match functional arg types "
            f"positionally with the first arg promoted to Tensor(a!). "
            f"Got functional schema {f_schema} and in-place schema "
            f"{i_schema}."
        )

    expected_prefix = f_schema.name + "_"
    if not i_schema.name.startswith(expected_prefix):
        raise ValueError(
            f"Suspicious reinplace registration: "
            f"{functional_op} -> {inplace_op}. "
            f"In-place op name '{i_schema.name}' should start with "
            f"'{expected_prefix}'. This usually indicates a typo "
            f"(e.g. 'add -> sub_' instead of 'add -> add_')."
        )


def _derive_mutated_args(inplace_op: Any) -> Tuple[int, ...]:
    """Return the positional indices of args that the in-place op
    mutates, derived from the schema's `Tensor(a!)` annotations.

    Computed via ``output_to_aliased_input_map``: each return with a
    write alias points back to the input arg position carrying the
    same alias set. The mutated positions are the input indices that
    appear as values in that map.

    Raises `ValueError` if `inplace_op` has no write-aliased outputs
    that match an input (i.e. it isn't actually an in-place op). The
    schema is the source of truth — if a custom op truly mutates an
    arg, its schema must declare so via `Tensor(a!)` on both the
    mutated input and the corresponding return. Otherwise
    functionalization, memory planning, and autograd will all silently
    mishandle it.
    """
    schema = _op_schema(inplace_op)
    out_to_in = output_to_aliased_input_map(schema)
    indices = tuple(sorted(set(out_to_in.values())))
    if not indices:
        raise ValueError(
            f"{inplace_op} has no Tensor(a!) write-aliased args that "
            f"match a corresponding return. If this op truly mutates, "
            f"fix its schema to declare the mutated arg(s) with "
            f"`Tensor(a!)` on both the input and the matching return. "
            f"The schema is the contract that all of export, memory "
            f"planning, and functionalization rely on."
        )
    return indices


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _is_safe_to_reinplace(
    node: torch.fx.Node,
    later_nodes: Set[torch.fx.Node],
    inputs: Set[torch.fx.Node],
    mutable_inputs: Set[torch.fx.Node],
) -> bool:
    # This node is used later in the graph so we can't reinplace it
    # There is probably a faster way to do this but this works for now.
    if node in later_nodes:
        return False
    # If its not an input then we can reinplace it
    if node not in inputs:
        return True
    # If its a mutable input then we can reinplace it
    elif node in mutable_inputs:
        return True
    else:  # input but not mutable input
        return False


def _is_mutable_user_input(
    node: torch.fx.Node, exported_program: ExportedProgram
) -> bool:
    return (
        node.target in exported_program.graph_signature.user_inputs_to_mutate.values()
    )


def _is_mutable_buffer(node: torch.fx.Node, exported_program: ExportedProgram) -> bool:
    if node.target not in exported_program.graph_signature.inputs_to_buffers:
        return False
    buf = exported_program.graph_signature.inputs_to_buffers[node.target]
    return buf in exported_program.graph_signature.buffers_to_mutate.values()


# ---------------------------------------------------------------------------
# Pass entry point.
# ---------------------------------------------------------------------------


def reinplace_pass(  # noqa: C901
    ep: ExportedProgram,
    ops_to_inplace: Optional[Iterable[Any]] = None,
    inplace_overrides: Optional[Dict[Any, Any]] = None,
) -> ExportedProgram:
    """Rewrite functional ops in-place when safe.

    Walks the graph in reverse topological order. For each
    `call_function` node whose target is in the resolved set of
    in-place candidates, checks whether each mutated-arg position is
    safe to reinplace (`_is_safe_to_reinplace`). If all checks pass,
    replaces the node with a call to its in-place op variant.

    Safety rules:
      * The mutated arg must not be used by any later node in the
        graph.
      * If the mutated arg is a placeholder (program input), it must
        be a *mutable* input — i.e., declared in
        `graph_signature.user_inputs_to_mutate` or
        `graph_signature.buffers_to_mutate`. Immutable inputs are
        never reinplaced because mutating them would be a side effect
        the caller did not opt into.

    Args:
        ep: The ExportedProgram to rewrite. Modified in place; returned
            for chaining.
        ops_to_inplace: Optional iterable of functional edge-dialect
            ops the pass should try to reinplace. The in-place form is
            auto-derived from the schema (`X -> X_*` by name + schema
            match). Defaults to `DEFAULT_INPLACEABLE_OPS`. Pass an
            empty iterable to disable all rewrites:

                from executorch.exir.dialects._ops import ops

                reinplace_pass(
                    ep,
                    ops_to_inplace=DEFAULT_INPLACEABLE_OPS | {
                        ops.edge.aten.index_copy.default,
                    },
                )

        inplace_overrides: Optional explicit map for non-conventional
            rewrites (e.g. backend-fused in-place ops whose name does
            not follow the `X -> X_*` convention). Keys are also
            included in the set of ops to consider — you do NOT need
            to also list them in `ops_to_inplace`. Example:

                reinplace_pass(
                    ep,
                    inplace_overrides={
                        my_backend.functional: my_backend.fused_inplace,
                    },
                )

        Each entry is validated at pass startup:
          * Schema arg types must match positionally between functional
            and in-place forms.
          * The in-place op's first arg must carry `Tensor(a!)`.
          * The in-place op's name must start with the functional op's
            name + "_" (e.g. `aten::add -> aten::add_*`).
          * The in-place op must have at least one `Tensor(a!)` arg.
        Misregistrations raise `ValueError` immediately.

    Returns:
        The (possibly mutated) ExportedProgram.
    """
    overrides = inplace_overrides or {}
    op_set: Set[Any] = set(
        ops_to_inplace if ops_to_inplace is not None else DEFAULT_INPLACEABLE_OPS
    )
    # Overrides also enroll their key in the candidate set.
    op_set.update(overrides.keys())

    # Validate every entry up front and pre-compute mutated_args so we
    # don't re-do the schema introspection per node.
    resolved: Dict[Any, Tuple[Any, Tuple[int, ...]]] = {}
    for functional_op in op_set:
        if functional_op in overrides:
            inplace_op = overrides[functional_op]
        else:
            inplace_op = _derive_edge_inplace_overload(functional_op)
            if inplace_op is None:
                raise ValueError(
                    f"Cannot auto-derive in-place form for "
                    f"{functional_op}. Provide an explicit mapping via "
                    f"`inplace_overrides={{{functional_op}: <inplace_op>}}`."
                )
        _validate_inplace_mapping(functional_op, inplace_op)
        mutated_args = _derive_mutated_args(inplace_op)
        resolved[functional_op] = (inplace_op, mutated_args)

    seen_nodes: Set[torch.fx.Node] = set()
    # Get all placeholders
    inputs: Set[torch.fx.Node] = set()
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            inputs.add(node)
    # Get all inputs that we could potentially mutate
    mutable_nodes: Set[torch.fx.Node] = {
        node
        for node in inputs
        if _is_mutable_user_input(node, ep) or _is_mutable_buffer(node, ep)
    }

    for node in reversed(ep.graph.nodes):
        entry = resolved.get(node.target) if node.op == "call_function" else None
        if entry is not None:
            inplace_op, mutated_args = entry
            # Every mutated arg position must independently be safe.
            all_safe = True
            for arg_idx in mutated_args:
                if arg_idx >= len(node.args):
                    raise ValueError(
                        f"reinplace: {node.target} call at {node} has "
                        f"{len(node.args)} positional args, but the "
                        f"schema declares position {arg_idx} as "
                        f"Tensor(a!). Export should normalize mutated "
                        f"args to positional; this graph violates that "
                        f"assumption."
                    )
                arg_node = node.args[arg_idx]
                if not isinstance(arg_node, torch.fx.Node):
                    raise ValueError(
                        f"reinplace: {node.target} call at {node} has a "
                        f"non-Node value {arg_node!r} at position "
                        f"{arg_idx}, but the schema declares it as "
                        f"Tensor(a!). A Tensor input in an FX graph "
                        f"must be a torch.fx.Node."
                    )
                if not _is_safe_to_reinplace(
                    arg_node, seen_nodes, inputs, mutable_nodes
                ):
                    all_safe = False
                    break
            if all_safe:
                with ep.graph.inserting_before(node):
                    # Forward both args and kwargs: the in-place overload
                    # is schema-matched to the functional one, so any
                    # kwarg valid on the functional op (e.g.
                    # `accumulate=` for `index_put`) is also valid on
                    # the in-place form. Dropping kwargs would silently
                    # change semantics.
                    new_node = ep.graph.call_function(
                        inplace_op,
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                    new_node.meta["val"] = node.meta["val"]
                    node.replace_all_uses_with(new_node)
                    ep.graph.erase_node(node)
                # No explicit `seen_nodes` update needed: the new
                # in-place node's target isn't in `op_set`, so the
                # reverse iterator visits it next and falls through
                # to the generic update below.
                continue
        # Note: this intentionally falls through for mapping-matched
        # nodes that failed the safety check. Their inputs *are* added
        # to seen_nodes, so further-upstream candidates correctly see
        # those tensors as "used later" and refuse to reinplace any op
        # that mutates them.
        # See test_unsafe_downstream_blocks_upstream_reinplace.
        if node.op == "call_function":
            seen_nodes.update(node.all_input_nodes)
    return ep
