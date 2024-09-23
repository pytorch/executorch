# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class TagIOQuantPass(ExportPass):
    """
    Pass run before partitioning to tag Q/DQ on any placeholder and output
    to ensure we don't greedily partition them for device. Float conversion
    has to happen outside a TOSA base inference profile.
    """

    def is_quant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        }

    def is_dequant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            # tag q of input
            if node.op == "placeholder":
                for user in node.users.keys():
                    # if we have an input going into a quantize
                    if self.is_quant_node(user):
                        user.meta["arm_override_partition"] = False

            # tag dq of outputs
            if node.op == "output":
                quant, *_ = node.args[0]
                if self.is_dequant_node(quant):
                    quant.meta["arm_override_partition"] = False

        graph_module.recompile()
        return PassResult(graph_module, True)
