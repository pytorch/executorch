# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from executorch.exir.graph_module import bfs_trace_with_node_process
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class DebugHandleGeneratorPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        """Lower a quantized reference model (with reference quantized operator patterns)
        to executorch backend, that has a canonical set of quantized operators
        """

        FROM_NODE_KEY = "from_node"
        DEBUG_HANDLE_KEY = "debug_handle"

        source_node_to_debug_handle: Dict[str, int] = {}

        def _get_greatest_ancestor_source_node(node: Node) -> str:
            """Get the source of the greatest ancestor node of the given node. The source
            here means the name of the node concated with the id the graph it belongs to.
            For example, if the node transformation is node a -> b -> c, then the greatest
            ancestor node of c is a.
            """

            node_source = node.meta[FROM_NODE_KEY]
            node_source = node_source[-1]

            while len(node_source.from_node) > 0:
                node_source = node_source.from_node[-1]

            return node_source.name + str(node_source.graph_id)

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

            source_node = _get_greatest_ancestor_source_node(node)

            debug_handle = (
                len(source_node_to_debug_handle) + 1
                if source_node not in source_node_to_debug_handle
                else source_node_to_debug_handle[source_node]
            )
            source_node_to_debug_handle[source_node] = debug_handle

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
