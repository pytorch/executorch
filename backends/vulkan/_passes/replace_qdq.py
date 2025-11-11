# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.vulkan.utils as utils
import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceQDQPass(ExportPass):
    """
    Replace standard quantize/dequantize ops with custom conv-specific ops when they
    feed into/from quantized convolution operations. This optimization allows the
    backend to handle quantization more efficiently for convolution operations.
    """

    def __init__(self):
        super(ReplaceQDQPass, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        # Track nodes that need to be replaced
        nodes_to_replace = []

        for node in graph_module.graph.nodes:
            # Check if this is the custom quantized conv2d op
            if node.target in [
                exir_ops.edge.et_vk.conv2d_q8ta_q8csw_q8to.default,
                exir_ops.edge.et_vk.conv2d_q8ta_q8csw_q8to_dw.default,
                exir_ops.edge.et_vk.add_q8ta_q8ta_q8to.default,
            ]:
                for quantized_input_node in node.args:
                    if isinstance(
                        quantized_input_node, torch.fx.Node
                    ) and utils.is_quant_node(quantized_input_node):
                        # Get the arguments from the original quantize node
                        input_tensor = quantized_input_node.args[0]
                        scale = quantized_input_node.args[1]
                        zero_point = quantized_input_node.args[2]

                        nodes_to_replace.append(
                            {
                                "old_node": quantized_input_node,
                                "new_target": exir_ops.edge.et_vk.quantize_q8ta_for_conv2d.default,
                                "args": (input_tensor, scale, zero_point),
                                "node_type": "quantize_input",
                            }
                        )

                # Find dequantize ops that consume the output of this conv2d
                for user in node.users:
                    if utils.is_dequant_node(user):
                        # Get the arguments from the original dequantize node
                        scale = user.args[1]
                        zero_point = user.args[2]

                        nodes_to_replace.append(
                            {
                                "old_node": user,
                                "new_target": exir_ops.edge.et_vk.dequantize_q8to_from_conv2d.default,
                                "args": (
                                    node,
                                    scale,
                                    zero_point,
                                ),  # node is the conv2d output
                                "node_type": "dequantize_output",
                            }
                        )

        # Apply the replacements
        for replacement in nodes_to_replace:
            old_node = replacement["old_node"]
            new_target = replacement["new_target"]
            new_args = replacement["args"]

            with graph_module.graph.inserting_before(old_node):
                new_node = graph_module.graph.create_node(
                    "call_function", new_target, args=new_args
                )
                new_node.meta = old_node.meta.copy()
                old_node.replace_all_uses_with(new_node)

        # Clean up the graph
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Re-trace to validate everything is ok
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
