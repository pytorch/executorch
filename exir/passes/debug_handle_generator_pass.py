# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.graph_module import bfs_trace_with_node_process
from executorch.exir.pass_base import ExportPass
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

class DebugHandleGeneratorPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        """Lower a quantized reference model (with reference quantized operator patterns)
        to executorch backend, that has a canonical set of quantized operators
        """

        index = 1

        def _extract_debug_handles_from_node(node):
            nonlocal index
            node.meta["debug_handle"] = index
            index += 1

        bfs_trace_with_node_process(graph_module, _extract_debug_handles_from_node)

        return PassResult(graph_module, True)


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
