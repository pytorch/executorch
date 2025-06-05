# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class RemoveRedundantOpsPass(XNNPACKPass):
    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        original_nodes = list(graph.nodes)

        # Store first subsequent visitation of to_copy node
        prev = None
        for node in original_nodes:
            if len(node.all_input_nodes) == 0:
                continue

            # If we encounter a to_copy node, check if it is preceded by an opposite to_copy node
            if node.target == exir_ops.edge.aten._to_copy.default:
                if prev and ChannelsLastTaggedReshapePass.is_nchw_node(
                    prev
                ) != ChannelsLastTaggedReshapePass.is_nchw_node(node):
                    # If we find an opposite to_copy node, remove both nodes
                    prevPrev = prev.args[0]

                    for user in node.users.copy():
                        user.replace_input_with(node, prevPrev)

                    graph.erase_node(node)
                    graph.erase_node(prev)

                    prev = None
                    continue
                prev = node
            else:
                prev = None

        graph_module.recompile()

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
