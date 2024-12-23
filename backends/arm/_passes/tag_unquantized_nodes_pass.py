# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm.tosa_quant_utils import dq_q_ops, get_neighbour_quant_args
from executorch.exir.pass_base import ExportPass, PassResult


class TagUnquantizedNodesPass(ExportPass):
    """
    Pass run before partitioning to tag unquantized nodes
    to ensure we don't greedily partition them for device. Unquantized operations must remain on the CPU.
    """

    def is_node_quantized(self, node: torch.fx.Node) -> bool:
        user_q_args, input_q_args = get_neighbour_quant_args(node)

        # If there are no neighboring quantized nodes, then this node is not quantized except for constants,
        # they can only have a dequantization node.
        if (
            len(node.all_input_nodes) > 0
            and len(input_q_args) == 0
            or len(user_q_args) == 0
        ):
            return False

        return True

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            # Look through operations that are not quantization or dequantization
            if node.op == "call_function" and node.target not in dq_q_ops:
                is_node_quantized = self.is_node_quantized(node)
                if not is_node_quantized:
                    # For a non-quantized node, we tag the node and its inputs and outputs.
                    node.meta["arm_override_partition"] = False
                    for input_node in node.all_input_nodes:
                        input_node.meta["arm_override_partition"] = False
                    for user in node.users.keys():
                        user.meta["arm_override_partition"] = False

        graph_module.recompile()
        return PassResult(graph_module, True)
