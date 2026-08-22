# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Post-`to_backend` pass: replace each `executorch_devtools::tap.Tensor` node
with reducers.

This pass MUST run *after* `to_edge_transform_and_lower(...)` and *before*
`to_executorch()`. Running it before partitioning would defeat the whole
mechanism (the reducer ops would be eligible for delegation).
"""

from __future__ import annotations

import torch.fx as fx
from executorch.devtools.intermediate_output_tap._reducers import get_reducer
from executorch.devtools.intermediate_output_tap._tap_pass import find_tap_nodes
from torch.export.exported_program import OutputSpec, TensorArgument


def strip_taps_(edge_manager) -> None:
    """
    Replace every `tap.Tensor(src, reducer_name)` node in every method of
    `edge_manager` with the materialised reducer subgraph, in place.

    Args:
        edge_manager: An EdgeProgramManager (post-`to_edge_transform_and_lower`).
    """
    for method_name in edge_manager.methods:
        ep = edge_manager.exported_program(method_name)
        _strip_taps_in_exported_program(ep)


def _strip_taps_in_exported_program(ep) -> None:
    """Strip taps in a single ExportedProgram, in place."""
    gm = ep.graph_module
    graph = gm.graph
    tap_nodes = find_tap_nodes(gm)
    if not tap_nodes:
        return

    output_node = graph.output_node()

    for tap in tap_nodes:
        # tap.args = (src_node, reducer_name)
        src, reducer_name = tap.args[0], tap.args[1]
        reducer = get_reducer(str(reducer_name))

        with graph.inserting_before(tap):
            replacement = reducer.emit(graph, src)

        for use_node in list(tap.users.keys()):
            if use_node is output_node:
                new_outs = tuple(
                    replacement if a is tap else a for a in output_node.args[0]
                )
                output_node.args = (new_outs,)
            else:
                use_node.replace_input_with(tap, src)

    graph.eliminate_dead_code()
    gm.recompile()

    # Sync graph_signature.output_specs[i].arg.name to the (now-rewritten)
    # output node names. Without this, OutputSpec entries still reference the
    # old `tap.Tensor` node names that no longer exist in the graph, and the
    # ExecuTorch runtime crashes when wiring outputs by name.
    final_outs = output_node.args[0]
    output_specs = ep.graph_signature.output_specs
    assert len(final_outs) == len(output_specs), (
        f"output_node arg count {len(final_outs)} != output_specs count "
        f"{len(output_specs)} — EP is malformed before strip"
    )
    for i, (spec, node) in enumerate(zip(output_specs, final_outs)):
        if not isinstance(node, fx.Node):
            continue
        if not isinstance(spec.arg, TensorArgument):
            continue
        if spec.arg.name == node.name:
            continue
        # Replace the whole OutputSpec entry — both OutputSpec and
        # TensorArgument may be frozen dataclasses, so build new instances.
        output_specs[i] = OutputSpec(
            kind=spec.kind,
            arg=TensorArgument(name=node.name),
            target=spec.target,
        )
