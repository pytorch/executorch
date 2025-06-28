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


class RemoveRedundantCopyPass(XNNPACKPass):
    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        original_nodes = list(graph.nodes)

        for node in original_nodes:
            if len(node.all_input_nodes) == 0:
                continue

            # If we encounter a to_copy node, check if its input is also a to_copy node with opposite format
            if node.target == exir_ops.edge.aten._to_copy.default:
                input_node = node.args[0]
                if (
                    input_node.target == exir_ops.edge.aten._to_copy.default
                    and ChannelsLastTaggedReshapePass.is_nchw_node(input_node)
                    != ChannelsLastTaggedReshapePass.is_nchw_node(node)
                    and len(input_node.users) == 1  # Ensure the first copy has no other users
                ):
                    # If we find an opposite to_copy node, remove both nodes
                    original_input = input_node.args[0]

                    for user in node.users.copy():
                        user.replace_input_with(node, original_input)

                    graph.erase_node(node)
                    graph.erase_node(input_node)

        graph_module.recompile()

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
