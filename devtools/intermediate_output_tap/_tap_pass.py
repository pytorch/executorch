# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
AOT pass: insert `tap.Tensor` placeholders after selected nodes and surface
them as additional USER_OUTPUTs of the ExportedProgram.

Pattern stolen from `executorch/exir/passes/weights_to_outputs_pass.py`:
- find existing output node
- build new output args (existing + new tap nodes)
- create new output node, replace_all_uses_with, erase old
- append OutputSpec(USER_OUTPUT) entries to gs.output_specs
- eliminate_dead_code() + recompile()
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.fx as fx
from executorch.devtools.intermediate_output_tap import (  # noqa: F401  registers tap.Tensor
    custom_ops_lib,
)
from executorch.devtools.intermediate_output_tap._reducers import (
    get_reducer,
    StatReducer,
    STATS,
)
from executorch.devtools.intermediate_output_tap._selectors import (
    NodeSelector,
    select_all_call_function,
)
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from torch.export import ExportedProgram
from torch.export.exported_program import OutputKind, OutputSpec, TensorArgument


# Public alias: one (selector, reducer) pair. A `rules=` argument is a
# sequence of these (or a single one as syntactic sugar).
TapRule = tuple[NodeSelector, "str | StatReducer"]


_TAP_TARGET = torch.ops.executorch_devtools.tap.Tensor


def is_tap_node(n: fx.Node) -> bool:
    return n.op == "call_function" and n.target is _TAP_TARGET


def tap_intermediate_outputs_(  # noqa: C901
    ep: ExportedProgram,
    rules: Sequence[TapRule] | TapRule | None = None,
    *,
    max_taps: int | None = None,
    error_on_empty: bool = True,
) -> tuple[ExportedProgram, list[TapSpec]]:
    """
    Rewrite `ep` IN PLACE so each node matching a rule's selector has its
    output appended to the program outputs (wrapped in a `tap.Tensor`
    placeholder that survives partitioning). Returns `ep` and the list of
    TapSpecs.

    Caller is responsible for `copy.deepcopy(ep)` first if they want to
    preserve the original.

    The returned EP is safe to feed to
    `to_edge_transform_and_lower(...).to_executorch()` *after* calling
    `strip_taps_(edge_manager)` to replace the placeholders with their
    reducer subgraphs (or identities, for FULL_TENSOR).

    Args:
        ep: The ExportedProgram to tap (mutated in place).
        rules: Sequence of `(selector, reducer)` pairs. Each candidate node
            is tested against the rules in order and tapped with the FIRST
            matching rule's reducer. A single `(selector, reducer)` tuple is
            also accepted as a shortcut for `[(selector, reducer)]`.
            Defaults to `[(select_all_call_function(), STATS)]` — i.e. tap
            every call_function node with the STATS reducer.
        max_taps: Optional cap on the number of taps inserted. Useful as a
            safety valve on very large graphs while iterating on rule
            patterns.
        error_on_empty: If True (default), raise `ValueError` when no nodes
            match any rule. Set to False to only emit a `UserWarning` and
            return `(ep, [])` — handy when iterating on rule patterns.
    """
    # Normalize `rules` to a non-empty list of (selector, reducer_obj) pairs.
    if rules is None:
        rules_list: Sequence[TapRule] = [(select_all_call_function(), STATS)]
    elif isinstance(rules, tuple) and len(rules) == 2 and callable(rules[0]):
        # Single `(selector, reducer)` tuple as syntactic sugar.
        rules_list = [rules]
    else:
        rules_list = rules
    if not rules_list:
        raise ValueError("tap_intermediate_outputs_: `rules` must be non-empty.")
    normalized_rules: list[tuple[NodeSelector, StatReducer]] = [
        (sel, get_reducer(red)) for sel, red in rules_list
    ]

    gs = ep.graph_signature
    gm = ep.graph_module
    graph = gm.graph
    output_node = graph.output_node()
    existing_outputs = list(output_node.args[0])

    # Snapshot before we start mutating the graph.
    candidate_nodes = [n for n in graph.nodes if not is_tap_node(n)]

    specs: list[TapSpec] = []
    new_tap_nodes: list[fx.Node] = []

    for node in candidate_nodes:
        if node.op != "call_function":
            continue
        # First-match wins.
        matched: StatReducer | None = None
        for sel, red in normalized_rules:
            if sel(node):
                matched = red
                break
        if matched is None:
            continue
        if max_taps is not None and len(specs) >= max_taps:
            break

        with graph.inserting_after(node):
            tap_node = graph.call_function(
                _TAP_TARGET,
                args=(node, matched.name),
            )
        # Copy standard FX provenance keys so torch's pretty-printers and
        # error messages can attribute the tap node back to its source op.
        if "from_node" in node.meta:
            tap_node.meta["from_node"] = node.meta["from_node"]
        if "stack_trace" in node.meta:
            tap_node.meta["stack_trace"] = node.meta["stack_trace"]
        if "nn_module_stack" in node.meta:
            tap_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]

        new_tap_nodes.append(tap_node)
        # Leaf module FQN + bare class from nn_module_stack (e.g.,
        # "layers.0.attention.wqs.0" / "Linear").
        module_path: str | None = None
        module_class: str | None = None
        stack = node.meta.get("nn_module_stack")
        if stack:
            try:
                last_entry = list(stack.values())[-1]
                if isinstance(last_entry, tuple):
                    module_path = last_entry[0]
                    if len(last_entry) >= 2:
                        mod_type = last_entry[1]
                        cls_name = getattr(mod_type, "__name__", None)
                        if cls_name is None:
                            cls_name = str(mod_type).rsplit(".", 1)[-1].rstrip("'>")
                        module_class = cls_name
                else:
                    module_path = str(last_entry)
            except Exception:
                module_path = None
                module_class = None
        specs.append(
            TapSpec(
                node_name=node.name,
                op_target=str(node.target),
                output_index=len(existing_outputs) + len(specs),
                reducer_name=matched.name,
                fields=matched.fields,
                stack_trace=node.meta.get("stack_trace"),
                module_path=module_path,
                module_class=module_class,
            )
        )

    if not new_tap_nodes:
        msg = (
            "tap_intermediate_outputs_: no rule matched any node. "
            "Double-check your rule predicates, "
            "or pass `error_on_empty=False` to suppress this error."
        )
        if error_on_empty:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return ep, []

    # Splice new outputs into the graph (mirror weights_to_outputs_pass).
    new_output_args = tuple(existing_outputs + new_tap_nodes)
    with graph.inserting_before(output_node):
        new_output = graph.output(new_output_args)
    output_node.replace_all_uses_with(new_output)
    graph.erase_node(output_node)

    # Append OutputSpec entries so the EP's signature matches the graph.
    for tap_node in new_tap_nodes:
        gs.output_specs.append(
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=tap_node.name),
                target=None,
            )
        )

    # Update each ModuleCallSignature's out_spec so `to_edge`'s re-trace can
    # unflatten the new flat output list. The "" (root) entry holds the
    # user-facing forward output structure; we wrap it in a tuple alongside
    # the new tap leaves and re-derive the spec.
    _extend_module_call_graph_outputs(ep, new_tap_nodes)

    graph.eliminate_dead_code()
    gm.recompile()
    return ep, specs


def _extend_module_call_graph_outputs(
    ep: ExportedProgram,
    new_tap_nodes: list[fx.Node],
) -> None:
    """
    Append `len(new_tap_nodes)` extra leaves to the root module-call entry's
    `out_spec` so the pytree unflatten step in `run_decompositions` works.
    Also extends the entry's `outputs: list[ArgumentSpec]`.

    NOTE: We append TensorArgument(name="") for each new tap output. Empty
    names are *skipped* by `_verify_exported_program_module_call_graph` (its
    check is `if arg.name and arg.name not in nodes`). We can't use the
    pre-trace tap node names because `to_edge`'s re-trace renames nodes via
    `from_node` chains, and our tap nodes' provenance wouldn't update them
    correctly — leading to "Output X does not exist in the graph" errors.
    The verifier's name check is metadata-only; the actual pytree unflatten
    only needs `out_spec` to have the correct number of leaves.
    """
    import torch.utils._pytree as pytree
    from torch.export.exported_program import TensorArgument as _TensorArgument

    n_new = len(new_tap_nodes)
    if n_new == 0:
        return

    for entry in ep._module_call_graph:
        if entry.fqn != "":
            continue
        sig = entry.signature
        if sig is None:
            continue
        old_spec = sig.out_spec
        # Build a dummy structure matching the old spec, then wrap with N new
        # leaves and re-derive the spec. This handles arbitrary pytree shapes.
        old_dummy = pytree.tree_unflatten([0] * old_spec.num_leaves, old_spec)
        if isinstance(old_dummy, tuple):
            new_dummy = (*old_dummy, *([0] * n_new))
        else:
            new_dummy = (old_dummy, *([0] * n_new))
        sig.out_spec = pytree.tree_structure(new_dummy)
        for _ in range(n_new):
            sig.outputs.append(_TensorArgument(name=""))
        break


def find_tap_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    """Helper: enumerate tap.Tensor nodes in a GraphModule (any dialect)."""
    out: list[fx.Node] = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        # Match across dialects:
        #   pre-edge:  torch.ops.executorch_devtools.tap.Tensor — str ends with name
        #   post-edge: <EdgeOpOverload: executorch_devtools.tap.Tensor>: schema = ...
        # so substring-match the qualified name.
        if "executorch_devtools.tap.Tensor" in str(n.target) or n.target is _TAP_TARGET:
            out.append(n)
    return out
