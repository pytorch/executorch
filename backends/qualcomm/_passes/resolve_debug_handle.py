# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator

import torch
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from executorch.exir.pass_base import ExportPass, PassResult


class ResolveDebugHandle(ExportPass):
    """
    Caution: This pass is executed as the last of the edge_passes.
    For any passes executed during qnn_preprocess, users will need to handle debug_handle ID themselves.

    Description: During passes transformation, some passes might be copying some node's meta when creating a new node,
    which means multiple nodes might be sharing the same debug_handle ID while it shouldn't.
    This is critical as Intermediate Debugger uses debug handle as key.
    debug_handle ID must be resolved so each op gets its own set of debug_handle ID and intermediate output.
    """

    def __init__(self):
        super(ResolveDebugHandle, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        handle_counter = 1
        visited = set()
        for node in graph_module.graph.nodes:
            # Assume node is traversed in topological order, adding a check here to be safe.
            if node.target == operator.getitem:
                source_node = node.args[0]
                assert (
                    source_node.name in visited
                ), "Graph is not traversed in topological order, unexpected behavior."
                node.meta[DEBUG_HANDLE_KEY] = source_node.meta[DEBUG_HANDLE_KEY]
            elif node.op == "call_function":
                node.meta[DEBUG_HANDLE_KEY] = handle_counter
                handle_counter += 1
            visited.add(node.name)

        graph_module.recompile()
        return PassResult(graph_module, True)
