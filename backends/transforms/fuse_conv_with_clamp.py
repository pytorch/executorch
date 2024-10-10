# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from executorch.backends.vulkan._passes.custom_ops_defs import (  # noqa
    conv_with_clamp_op,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class FuseClampPass(ExportPass):
    """
    Some activations like ReLU and hardtanh can be fused with certain operators (e.g. convolution) preceding it.
    """

    FUSEABLE_OPS = [
        exir_ops.edge.aten.convolution.default,
    ]
    FUSEABLE_ACTIVATIONS = [
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
    ]

    def get_output_min_max_from_activation(self, activation_node):
        if activation_node.target == exir_ops.edge.aten.relu.default:
            output_min = 0.0
            output_max = sys.float_info.max
        elif activation_node.target == exir_ops.edge.aten.hardtanh.default:
            output_min = -1.0
            output_max = 1.0
            if len(activation_node.args) > 1:
                output_min = activation_node.args[1]
                output_max = activation_node.args[2]

        return output_min, output_max

    def call(self, graph_module: torch.fx.GraphModule):
        for activation_node in graph_module.graph.nodes:
            if activation_node.op == "call_function":
                if activation_node.target in self.FUSEABLE_ACTIVATIONS:
                    preceding_op = activation_node.args[0]
                    if (
                        preceding_op.op == "call_function"
                        and preceding_op.target in self.FUSEABLE_OPS
                    ):
                        # Delete activation
                        output_min_max = self.get_output_min_max_from_activation(
                            activation_node
                        )
                        new_args = list(preceding_op.args)
                        new_args.append(output_min_max[0])
                        new_args.append(output_min_max[1])
                        new_args = tuple(new_args)
                        activation_node.replace_all_uses_with(preceding_op)
                        graph_module.graph.erase_node(activation_node)

                        # Create and insert node of custom op `conv_with_clamp`
                        with graph_module.graph.inserting_before(preceding_op):
                            conv_activation_node = graph_module.graph.create_node(
                                "call_function",
                                torch.ops.et_vk.conv_with_clamp.default,
                                new_args,
                            )

                            preceding_op.replace_all_uses_with(conv_activation_node)
                            graph_module.graph.erase_node(preceding_op)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
