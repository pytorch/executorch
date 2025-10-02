# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set

import torch
from executorch.exir.dialects._ops import ops
from torch.export import ExportedProgram


def _is_index_put(node: torch.fx.Node) -> bool:
    """Check if a node is an index_put operation."""
    return node.op == "call_function" and node.target in (
        torch.ops.aten.index_put.default,
        ops.edge.aten.index_put.default,
    )


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


def reinplace_pass(ep: ExportedProgram) -> ExportedProgram:
    """
    Pass that loops over nodes in an exported program and collects the first argument
    of every call_function node that is a view_copy operation.

    Args:
        exported_program: The ExportedProgram to analyze

    Returns:
        Set of nodes that are first arguments to view_copy operations
    """
    seen_nodes: Set[torch.fx.Node] = set()
    # Get all placeholders
    inputs = set()
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            inputs.add(node)
    # Get all inputs that we could potentially mutate
    mutable_nodes = set(
        [
            node
            for node in inputs
            if _is_mutable_user_input(node, ep) or _is_mutable_buffer(node, ep)
        ]
    )

    results = set()
    for node in reversed(ep.graph.nodes):
        if _is_index_put(node):
            # Check if this index_put node is safe to inplace
            # The first argument is the base tensor being indexed into
            first_arg = node.args[0]
            if _is_safe_to_reinplace(first_arg, seen_nodes, inputs, mutable_nodes):
                # This index_put is safe to reinplace
                with ep.graph.inserting_before(node):
                    new_node = ep.graph.call_function(
                        ops.edge.aten.index_put_.default, args=node.args
                    )
                    new_node.meta["val"] = node.meta["val"]
                    node.replace_all_uses_with(new_node)
                    ep.graph.erase_node(node)
                results.add(first_arg)
        elif node.op == "call_function":
            seen_nodes.update(node.all_input_nodes)
    return ep
