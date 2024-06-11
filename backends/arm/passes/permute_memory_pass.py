# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.passes.arm_pass import ArmPass
from executorch.backends.arm.tosa_quant_utils import q_op
from executorch.backends.arm.tosa_utils import (
    is_bias_node_for_quantized_addmm,
    is_consumer_node_depthwise_conv2d,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class PermuteMemoryPass(ArmPass):

    def is_weight_node_for_dw_conv(self, node: torch.fx.Node):
        if node.name in self.exported_program.graph_signature.inputs_to_parameters:
            consumer_node = list(node.users)[0]
            if not is_bias_node_for_quantized_addmm(node) and (
                not consumer_node.target == exir_ops.edge.aten.convolution.default
                or not list(consumer_node.users)[0].target == q_op
            ):
                return is_consumer_node_depthwise_conv2d(node)
        elif node.name in self.exported_program.graph_signature.inputs_to_buffers:
            return is_consumer_node_depthwise_conv2d(list(node.users)[0])
        elif node.op == "call_function" and is_consumer_node_depthwise_conv2d(node):
            if len(node.args) > 0:
                return (
                    node.args[0].name
                    in self.exported_program.graph_signature.inputs_to_buffers
                )
        return False

    def call(self, graph_module: torch.fx.GraphModule):
        NHWC_Order = (0, 2, 3, 1)
        HWCM_Order = (2, 3, 0, 1)
        for node in graph_module.graph.nodes:
            if type(node.meta["val"]) is tuple:
                data = node.meta["val"][0].data
            else:
                data = node.meta["val"].data

            if len(data.shape) == 4:
                dim_order = NHWC_Order
                if self.is_weight_node_for_dw_conv(node):
                    dim_order = HWCM_Order
            else:
                dim_order = tuple(range(len(data.shape)))
            node.meta["tosa_dim_order"] = dim_order
        graph_module.recompile()
        return PassResult(graph_module, True)
