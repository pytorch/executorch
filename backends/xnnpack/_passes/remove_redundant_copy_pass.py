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
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class RemoveRedundantCopyPass(XNNPACKPass):
    def _safe_remove_node(self, node, graph):
        if len(node.users) == 0:
            graph.erase_node(node)

    def _try_remove_regular_redundant_to_copy(self, node, graph):
        """
        Try to remove redundant regular to_copy operations with pattern to_copy1 -> to_copy2 with opposite memory formats
        """
        input_node = node.args[0]

        # Check if input is a to_copy with opposite memory format
        if (
            input_node.target == exir_ops.edge.aten._to_copy.default
            and ChannelsLastTaggedReshapePass.is_nchw_node(input_node)
            != ChannelsLastTaggedReshapePass.is_nchw_node(node)
            and len(input_node.users) == 1
        ):  # Ensure the first copy has no other users

            # Get the original input (before the first to_copy)
            original_input = input_node.args[0]

            # Replace all users of the second to_copy with the original input
            for user in node.users.copy():
                user.replace_input_with(node, original_input)

            # Remove both to_copy nodes
            self._safe_remove_node(node, graph)
            self._safe_remove_node(input_node, graph)

            return True
        elif (
            ChannelsLastTaggedReshapePass.is_nhwc_node(input_node)
            and ChannelsLastTaggedReshapePass.is_nhwc_node(node)
        ) or (
            ChannelsLastTaggedReshapePass.is_nchw_node(input_node)
            and ChannelsLastTaggedReshapePass.is_nchw_node(node)
        ):
            # Replace all users of the second to_copy with the original input
            for user in node.users.copy():
                user.replace_input_with(node, input_node)
            self._safe_remove_node(node, graph)
            return True

        return False

    def _try_remove_quantized_redundant_to_copy(self, node, graph):
        """
        Try to remove redundant to_copy operations in quantized graphs with pattern dq1 -> to_copy1 -> q1 -> dq2 -> to_copy2 -> q2
        """
        # Check if this to_copy is followed by a quantize node
        if len(node.users) != 1:
            return False
        q_node = next(iter(node.users))
        if not is_quant(q_node):
            return False

        # Check if this to_copy is preceded by a dequantize node
        dq_node = node.args[0]
        if not is_dequant(dq_node):
            return False

        # Get the input to the dequantize node
        if len(dq_node.all_input_nodes) != 1:
            return False

        prev_q_node = dq_node.args[0]

        # Check if there's another dequantize -> to_copy -> quantize chain
        if not is_quant(prev_q_node) or len(prev_q_node.all_input_nodes) != 1:
            return False

        # Check if there's a to_copy before the previous quantize
        prev_to_copy = prev_q_node.args[0]
        if (
            prev_to_copy.target == exir_ops.edge.aten._to_copy.default
            and ChannelsLastTaggedReshapePass.is_nchw_node(prev_to_copy)
            != ChannelsLastTaggedReshapePass.is_nchw_node(node)
            and len(prev_to_copy.users) == 1
        ):  # Ensure the first copy has no other users
            prev_dq_node = prev_to_copy.args[0]
            if not is_dequant(prev_dq_node) or len(prev_dq_node.all_input_nodes) != 1:
                return False

            # Get the original input (before the first to_copy)
            original_input = prev_dq_node.args[0]

            # Replace all users of the second to_copy with the original input
            for user in q_node.users.copy():
                user.replace_input_with(q_node, original_input)

            # Remove nodes safely (only if they have no other users)
            self._safe_remove_node(q_node, graph)
            self._safe_remove_node(node, graph)
            self._safe_remove_node(dq_node, graph)
            self._safe_remove_node(prev_q_node, graph)
            self._safe_remove_node(prev_to_copy, graph)
            self._safe_remove_node(prev_dq_node, graph)
        elif (
            ChannelsLastTaggedReshapePass.is_nhwc_node(prev_to_copy)
            and ChannelsLastTaggedReshapePass.is_nhwc_node(node)
        ) or (
            ChannelsLastTaggedReshapePass.is_nchw_node(prev_to_copy)
            and ChannelsLastTaggedReshapePass.is_nchw_node(node)
        ):
            # Remove node and the q/dq around it only
            # Get the original quantized tensor (input to dq_node)
            original_q_tensor = dq_node.args[0]

            # Replace all users of q_node with the original quantized tensor
            for user in q_node.users.copy():
                user.replace_input_with(q_node, original_q_tensor)

            self._safe_remove_node(q_node, graph)
            self._safe_remove_node(node, graph)
            self._safe_remove_node(dq_node, graph)
            return True

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        original_nodes = list(graph.nodes)

        for node in original_nodes:
            if len(node.all_input_nodes) == 0:
                continue

            # Only process to_copy nodes
            if node.target != exir_ops.edge.aten._to_copy.default:
                continue

            if is_dequant(node.args[0]):
                self._try_remove_quantized_redundant_to_copy(node, graph)
            else:
                self._try_remove_regular_redundant_to_copy(node, graph)

        graph_module.recompile()

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
