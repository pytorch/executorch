# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from executorch.exir.debug_handle_utils import (
    DEBUG_HANDLE_KEY,
    FROM_NODE_KEY,
    get_greatest_ancestor_node_identifier,
)
from executorch.exir.graph_module import bfs_trace_with_node_process
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class DebugHandleGeneratorPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        """Generate debug handles for each node in the graph module and its submodule except
        placeholder and output nodes. The debug handle is generated starting from 1 and
        incrementally. The debug handle of a node is the same as the node sharing the same
        greatest ancestor node in the export flow.
        """

        source_node_id_to_debug_handle: Dict[str, int] = {}

        def _extract_debug_handles_from_node(node: Node) -> None:
            """
            Generate a debug handle based on node's oldest ancestor node's name
            and graph id, or return None if the node does not need to be traced.
            """

            if node.op == "placeholder" or node.op == "output":
                # placeholder and output nodes don't have debug handle
                return

            assert (
                FROM_NODE_KEY in node.meta
            ), f"Node {node} does not have meta key {FROM_NODE_KEY}"

            greatest_ancestor_node_id = get_greatest_ancestor_node_identifier(node)

            debug_handle = (
                len(source_node_id_to_debug_handle) + 1
                if greatest_ancestor_node_id not in source_node_id_to_debug_handle
                else source_node_id_to_debug_handle[greatest_ancestor_node_id]
            )

            source_node_id_to_debug_handle[greatest_ancestor_node_id] = debug_handle
            node.meta[DEBUG_HANDLE_KEY] = debug_handle

        bfs_trace_with_node_process(graph_module, _extract_debug_handles_from_node)

        return PassResult(graph_module, True)


# TODO(gasoonjia): generate missing debug handles using `from_node` info
def generate_missing_debug_handles(ep: ExportedProgram):
    """
    This pass is used to generate missing debug handles for the graph module and its submodules.
    """

    max_handle = 0

    def _extract_max_debug_handle(node):
        nonlocal max_handle
        if "debug_handle" in node.meta:
            max_handle = max(max_handle, node.meta["debug_handle"])

    def _insert_new_debug_handles(node):
        nonlocal max_handle
        if node.meta.get("debug_handle", 0) in (0, None):
            node.meta["debug_handle"] = max_handle + 1
            max_handle += 1

    bfs_trace_with_node_process(ep.graph_module, _extract_max_debug_handle)
    bfs_trace_with_node_process(ep.graph_module, _insert_new_debug_handles)
