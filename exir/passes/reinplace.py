# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
from executorch.exir.dialects._ops import ops
from torch.export import ExportedProgram


# ---------------------------------------------------------------------------
# Public API for extending the pass with additional ops.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InplaceTarget:
    """Describes how to rewrite a functional op into its in-place form.

    Attributes:
        inplace_op: The in-place op overload to rewrite to (e.g.
            ``torch.ops.aten.index_put_.default`` or its edge-dialect
            equivalent).
        mutated_args: Tuple of argument indices that the in-place op
            mutates. The pass only marks the rewrite as safe if every
            arg at these positions passes the safety check
            (`_is_safe_to_reinplace`). Defaults to ``(0,)`` which
            matches the historical behavior of treating the first
            positional arg as the mutated one.
    """

    inplace_op: Any
    mutated_args: Tuple[int, ...] = (0,)


# Default registry — preserves the historical behavior of this pass exactly.
# Today the pass only rewrites `index_put -> index_put_` (both aten and edge
# dialect variants). Callers that want to extend the pass to additional ops
# should pass `ops_to_inplace=` rather than mutating this dict.
DEFAULT_INPLACEABLE_OPS: Dict[Any, InplaceTarget] = {
    torch.ops.aten.index_put.default: InplaceTarget(
        inplace_op=torch.ops.aten.index_put_.default,
    ),
    ops.edge.aten.index_put.default: InplaceTarget(
        inplace_op=ops.edge.aten.index_put_.default,
    ),
}


# ---------------------------------------------------------------------------
# Schema-based in-place discovery.
#
# Many functional ops have an in-place counterpart whose overload name
# DIFFERS from the functional one. Examples in aten:
#   pow.Tensor_Scalar(Tensor, Scalar)   ↔ pow_.Scalar(Tensor(a!), Scalar)
# Simple name matching (`getattr(aten.pow_, "Tensor_Scalar")`) misses
# these. Schema matching does not — the in-place op has the same input
# arg types as the functional, with the first arg promoted to
# `Tensor(a!)` (write alias). The helpers below let backends build
# in-place registries by schema rather than by guessing names.
# ---------------------------------------------------------------------------


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
    # First arg of candidate must mutate (Tensor(a!)).
    if not c_args:
        return False
    first = c_args[0]
    if first.alias_info is None or not first.alias_info.is_write:
        return False
    # All arg types must match positionally.
    for fa, ca in zip(f_args, c_args):
        if str(fa.type) != str(ca.type):
            return False
    return True


def find_inplace_overload(functional_op: Any) -> Optional[Any]:
    """Given a functional op (`torch.ops.aten.<name>.<overload>`), find
    its in-place counterpart by walking `torch.ops.aten.<name>_.*`
    overloads and returning the one whose schema matches positionally
    (with the first arg promoted to `Tensor(a!)`).

    Returns None if no schema match is found, or if the op isn't an
    aten op (e.g., custom ops without an `_` package).

    Handles cross-overload-name asymmetry: e.g.,
        `aten.pow.Tensor_Scalar` → `aten.pow_.Scalar`
    where simple name-based lookup `aten.pow_.Tensor_Scalar` fails.
    """
    schema = getattr(functional_op, "_schema", None)
    if schema is None:
        return None
    # Schema name format: "aten::pow" or "namespace::op".
    name = schema.name
    namespace, base = (name.split("::", 1) + [""])[:2] if "::" in name else ("aten", name)
    if namespace != "aten":
        return None
    inplace_pkg = getattr(torch.ops.aten, base + "_", None)
    if inplace_pkg is None:
        return None
    for overload_name in inplace_pkg.overloads():
        candidate = getattr(inplace_pkg, overload_name, None)
        if candidate is None:
            continue
        cand_schema = getattr(candidate, "_schema", None)
        if cand_schema is None:
            continue
        if _is_inplace_of(schema, cand_schema):
            return candidate
    return None


def build_inplace_registry_for(
    functional_ops: "Iterable[Any]",
) -> Dict[Any, InplaceTarget]:
    """Build an `InplaceTarget` registry for the given functional ops by
    schema-matching against aten in-place overloads.

    Ops that have no schema-matching in-place counterpart are silently
    skipped — caller is responsible for deciding what to do about them
    (e.g., adding an explicit override entry).
    """
    registry: Dict[Any, InplaceTarget] = {}
    for op in functional_ops:
        inplace = find_inplace_overload(op)
        if inplace is not None:
            registry[op] = InplaceTarget(inplace_op=inplace)
    return registry


# ---------------------------------------------------------------------------
# Internal helpers (unchanged behavior).
# ---------------------------------------------------------------------------


def _lookup_inplace_target(
    node: torch.fx.Node, registry: Dict[Any, InplaceTarget]
) -> Optional[InplaceTarget]:
    """Return the InplaceTarget for `node`, or None if not in the registry."""
    if node.op != "call_function":
        return None
    return registry.get(node.target)


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


def reinplace_pass(
    ep: ExportedProgram,
    ops_to_inplace: Optional[Dict[Any, InplaceTarget]] = None,
) -> ExportedProgram:
    """Rewrite functional ops in-place when safe.

    Walks the graph in reverse topological order. For each
    `call_function` node whose target is in the `ops_to_inplace`
    registry, checks whether each mutated-arg position is safe to
    reinplace (`_is_safe_to_reinplace`). If all checks pass, replaces
    the node with a call to the registered in-place op variant.

    Safety rules (unchanged from the original pass):
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
        ops_to_inplace: Optional registry of `{functional_op:
            InplaceTarget}` entries. When omitted (`None`), the pass
            uses `DEFAULT_INPLACEABLE_OPS`, which preserves the
            historical behavior (only `index_put -> index_put_`).
            Callers that want to extend the pass to additional ops
            should construct their own registry, typically by merging
            with the default:

                my_registry = {
                    **DEFAULT_INPLACEABLE_OPS,
                    aten.index_copy.default: InplaceTarget(
                        aten.index_copy_.default
                    ),
                }
                reinplace_pass(ep, ops_to_inplace=my_registry)

            Passing an empty dict disables all rewrites.

    Returns:
        The (possibly mutated) ExportedProgram.
    """
    registry = (
        ops_to_inplace if ops_to_inplace is not None else DEFAULT_INPLACEABLE_OPS
    )

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
        target = _lookup_inplace_target(node, registry)
        if target is not None:
            # Every mutated arg position must independently be safe.
            all_safe = True
            for arg_idx in target.mutated_args:
                if arg_idx >= len(node.args):
                    all_safe = False
                    break
                arg_node = node.args[arg_idx]
                if not isinstance(arg_node, torch.fx.Node):
                    all_safe = False
                    break
                if not _is_safe_to_reinplace(
                    arg_node, seen_nodes, inputs, mutable_nodes
                ):
                    all_safe = False
                    break
            if all_safe:
                with ep.graph.inserting_before(node):
                    new_node = ep.graph.call_function(
                        target.inplace_op, args=node.args
                    )
                    new_node.meta["val"] = node.meta["val"]
                    node.replace_all_uses_with(new_node)
                    ep.graph.erase_node(node)
                # Keep behavior identical: only the rewritten node's
                # inputs are *not* added to seen_nodes (they're the
                # mutated args, not "later usages" of someone else's
                # output). This matches the original pass.
                continue
        if node.op == "call_function":
            seen_nodes.update(node.all_input_nodes)
    return ep
