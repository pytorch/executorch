# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Post-`to_backend` pass: replace each `executorch_devtools::tap.Tensor` node
with either an identity edge (FULL_TENSOR) or a portable reducer subgraph
(STATS, or any user-supplied StatReducer).

Pattern stolen from `remove_graph_break_` in
`executorch/examples/apple/coreml/llama/export_static_llm_coreml.py`.

This pass MUST run *after* `to_edge_transform_and_lower(...)` and *before*
`to_executorch()`. Running it before partitioning would defeat the whole
mechanism (the reducer ops would be eligible for delegation).

When called with the `tap_specs` from `tap_intermediate_outputs`, this pass
also populates `TapSpec.reducer_node_name` for each spec — the FX node name
of the post-strip reducer terminal. This is the bridge
`Inspector.calculate_numeric_gap_from_taps` uses to recover the
post-ETRecord-roundtrip `debug_handle` for alignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as _dataclass_replace

import torch.fx as fx
from executorch.devtools.intermediate_output_tap._reducers import get_reducer
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from executorch.devtools.intermediate_output_tap._tap_pass import find_tap_nodes


def strip_taps_(
    edge_manager,
    tap_specs: Sequence[TapSpec] | None = None,
) -> list[TapSpec] | None:
    """
    Replace every `tap.Tensor(src, reducer_name, debug_handle)` node in every
    method of `edge_manager` with the materialised reducer subgraph, in place.

    For FULL_TENSOR the placeholder is collapsed (the source node's value
    flows directly to whatever consumed the placeholder).

    Args:
        edge_manager: An EdgeProgramManager (post-`to_edge_transform_and_lower`).
        tap_specs: Optional. If provided, the pass returns a NEW list of
            TapSpecs with `reducer_node_name` populated for each spec — the
            FX name of the post-strip reducer terminal node. This list must
            be passed to `Inspector.calculate_numeric_gap_from_taps` for
            alignment to work.

    Returns:
        Updated tap_specs list if `tap_specs` was provided, else None.
    """
    # Walk in graph order; tap nodes appear in the same order they were
    # created by `tap_intermediate_outputs`, which is the same order as
    # `tap_specs`. Track each tap's replacement node so we can update the
    # corresponding spec.
    replacement_names: list[str | None] = []
    for method_name in edge_manager.methods:
        ep = edge_manager.exported_program(method_name)
        gm = ep.graph_module
        for replacement_node in _strip_taps_in_graph_module(gm):
            replacement_names.append(
                replacement_node.name if replacement_node is not None else None
            )

    if tap_specs is None:
        return None

    if len(tap_specs) != len(replacement_names):
        raise RuntimeError(
            f"strip_taps_: tap_specs length ({len(tap_specs)}) does not match "
            f"the number of tap nodes found in the edge_manager "
            f"({len(replacement_names)}). The strip pass cannot align specs "
            f"to reducer nodes. Did you call strip_taps_ on a different "
            f"edge_manager than the one produced from the tapped EP?"
        )

    return [
        _dataclass_replace(spec, reducer_node_name=name)
        for spec, name in zip(tap_specs, replacement_names)
    ]


def _strip_taps_in_graph_module(gm: fx.GraphModule) -> list[fx.Node | None]:
    """
    Strip taps in a single GraphModule. Returns the list of replacement nodes
    in tap-creation order (same as graph order). For FULL_TENSOR taps the
    "replacement" is the source node itself (since the tap collapses to
    identity).
    """
    graph = gm.graph
    tap_nodes = find_tap_nodes(gm)
    if not tap_nodes:
        return []

    output_node = graph.output_node()
    replacements: list[fx.Node | None] = []

    # Compute next available debug_handle so each reducer terminal gets a
    # unique one (necessary so Inspector can look it up by node name and find
    # a non-None handle in the post-roundtrip graph).
    existing_handles = [
        n.meta.get("debug_handle")
        for n in graph.nodes
        if isinstance(n.meta.get("debug_handle"), int)
    ]
    next_handle = (max(existing_handles) + 1) if existing_handles else 1

    for tap in tap_nodes:
        # tap.args = (src_node, reducer_name, debug_handle)
        src, reducer_name, dh = tap.args[0], tap.args[1], tap.args[2]
        reducer = get_reducer(str(reducer_name))

        if reducer.name == "FULL_TENSOR":
            # Identity: re-route all consumers to the source. The "reducer
            # terminal" is the source itself.
            tap.replace_all_uses_with(src)
            replacements.append(src if isinstance(src, fx.Node) else None)
            continue

        # Build the reducer subgraph (reads from src).
        with graph.inserting_before(tap):
            replacement = reducer.emit(graph, src)
        # Always assign a debug_handle to the reducer terminal so Inspector
        # can find it post-roundtrip. Prefer the source's pre-tap handle if
        # available (carries semantic meaning); otherwise use next_handle.
        if dh:
            replacement.meta["debug_handle"] = dh
        else:
            replacement.meta["debug_handle"] = next_handle
            next_handle += 1
        replacement.meta["is_tap"] = True
        replacement.meta["source_node"] = src.name if isinstance(src, fx.Node) else None

        # `tap` may have ended up in the data path during to_edge's re-trace
        # (because CompositeExplicitAutograd preserves the op as an identity
        # node, and re-traced consumers point at it instead of `src`). So:
        #   - the OUTPUT-node use becomes the reducer (the value we want
        #     surfaced as a tap).
        #   - every OTHER use is rewritten back to `src` (identity passthrough),
        #     restoring the original data path.
        for use_node in list(tap.users.keys()):
            if use_node is output_node:
                new_outs = tuple(
                    replacement if a is tap else a for a in output_node.args[0]
                )
                output_node.args = (new_outs,)
            else:
                use_node.replace_input_with(tap, src)
        replacements.append(replacement)

    graph.eliminate_dead_code()
    gm.recompile()
    return replacements
