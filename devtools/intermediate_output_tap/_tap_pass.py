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

import copy
from collections.abc import Callable

import torch
import torch.fx as fx
from executorch.devtools.intermediate_output_tap import custom_ops_lib  # noqa: F401  registers tap.Tensor
from executorch.devtools.intermediate_output_tap._reducers import (
    DEFAULT_STATS,
    get_reducer,
    StatReducer,
)
from executorch.devtools.intermediate_output_tap._selectors import (
    NodeSelector,
    select_all_call_function,
)
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from torch.export import ExportedProgram
from torch.export.exported_program import OutputKind, OutputSpec, TensorArgument


# Don't ever tap our own tap nodes if a user runs the pass twice.
# `tap.Tensor` is already an OpOverload (not a packet) since "Tensor" is the
# overload name — same convention as torch.ops.executorch_utils.graph_break.Tensor.
_TAP_TARGET = torch.ops.executorch_devtools.tap.Tensor


def _is_tap_node(n: fx.Node) -> bool:
    return n.op == "call_function" and n.target is _TAP_TARGET


def tap_intermediate_outputs(
    ep: ExportedProgram,
    selector: NodeSelector | None = None,
    reducer: str | StatReducer = DEFAULT_STATS,
    *,
    tap_name_prefix: str = "tap_",
    skip_if_no_debug_handle: bool = False,
    max_taps: int | None = None,
    inplace: bool = False,
) -> tuple[ExportedProgram, list[TapSpec]]:
    """
    Rewrite `ep` so each node matching `selector` has its output appended to
    the program outputs (wrapped in a `tap.Tensor` placeholder that survives
    partitioning). Returns the new ExportedProgram and a list of TapSpecs.

    The returned EP is safe to feed to
    `to_edge_transform_and_lower(...).to_executorch()` *after* calling
    `strip_taps_(edge_manager)` to replace the placeholders with their
    reducer subgraphs (or identities, for FULL_TENSOR).

    Args:
        ep: The ExportedProgram to tap.
        selector: A predicate over fx.Node. Defaults to
            `select_all_call_function()`. Tap nodes themselves are always
            excluded so re-running the pass is idempotent.
        reducer: Either a built-in reducer name ("DEFAULT_STATS",
            "MIN_MAX_MEAN", "ABS_MAX_ONLY", "FULL_TENSOR") or a custom
            StatReducer instance.
        tap_name_prefix: Prefix for the tap nodes' names. Helps when
            grepping the dumped graph.
        skip_if_no_debug_handle: If True, only tap nodes that already
            carry `node.meta["debug_handle"]`. Recommended for Inspector
            integration since handle-less taps cannot be aligned with
            AOT outputs.
        max_taps: Optional cap on number of taps. Helps avoid OOM for
            very large models.
        inplace: If False (default), deep-copy `ep` before mutating.
    """
    if selector is None:
        selector = select_all_call_function()
    reducer_obj = get_reducer(reducer)

    if not inplace:
        ep = copy.deepcopy(ep)

    gs = ep.graph_signature
    gm = ep.graph_module
    graph = gm.graph
    output_node = graph.output_node()
    existing_outputs = list(output_node.args[0])

    # Snapshot before we start mutating the graph.
    candidate_nodes = [n for n in graph.nodes if not _is_tap_node(n)]

    specs: list[TapSpec] = []
    new_tap_nodes: list[fx.Node] = []

    for node in candidate_nodes:
        if node.op != "call_function" or not selector(node):
            continue
        debug_handle = node.meta.get("debug_handle")
        if skip_if_no_debug_handle and debug_handle is None:
            continue
        if max_taps is not None and len(specs) >= max_taps:
            break

        # tap.Tensor's int arg cannot be None; sentinel 0 means "no handle".
        dh_arg = int(debug_handle) if isinstance(debug_handle, int) else 0

        with graph.inserting_after(node):
            tap_node = graph.call_function(
                _TAP_TARGET,
                args=(node, reducer_obj.name, dh_arg),
            )
        # Don't override the auto-assigned name — FX guarantees uniqueness.
        # Stash the prefixed-source-name in meta for human-readable logs.
        tap_node.meta["tap_label"] = f"{tap_name_prefix}{node.name}"
        # Preserve provenance for Inspector's `propagate_back_debug_handle`
        # and for users that pretty-print the graph.
        if debug_handle is not None:
            tap_node.meta["debug_handle"] = debug_handle
        if "from_node" in node.meta:
            tap_node.meta["from_node"] = node.meta["from_node"]
        if "stack_trace" in node.meta:
            tap_node.meta["stack_trace"] = node.meta["stack_trace"]
        if "nn_module_stack" in node.meta:
            tap_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
        tap_node.meta["is_tap"] = True
        tap_node.meta["source_node"] = node.name

        new_tap_nodes.append(tap_node)
        specs.append(
            TapSpec(
                node_name=node.name,
                op_target=str(node.target),
                debug_handle=debug_handle if isinstance(debug_handle, int) else None,
                output_index=len(existing_outputs) + len(specs),
                reducer_name=reducer_obj.name,
                fields=reducer_obj.fields,
                stack_trace=node.meta.get("stack_trace"),
            )
        )

    if not new_tap_nodes:
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


# Re-export the predicate so callers can identify tap nodes without importing
# torch.ops directly.
is_tap_node: Callable[[fx.Node], bool] = _is_tap_node
