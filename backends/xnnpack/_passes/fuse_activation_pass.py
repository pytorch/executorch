# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import OutputMinMax

from executorch.backends.xnnpack.utils.utils import check_or_raise
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class FuseActivationPass(XNNPACKPass):
    """
    Some activations like ReLU and hardtanh can be fused with certain operators preceding it.
    In the case of fusion, we can instead delete the relu node and embed the activation constraints in the metadata
    of the preceding node.
    """

    FUSED_ACTIVATION_TAG = "XNN_FUSED_ACTIVATION"

    FUSEABLE_OPS = [
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.linear.default,
    ]
    FUSEABLE_ACTIVATIONS = [
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
    ]

    @staticmethod
    def get_fused_activation(node):
        if node.meta.get(FuseActivationPass.FUSED_ACTIVATION_TAG, None) is not None:
            return node.meta[FuseActivationPass.FUSED_ACTIVATION_TAG]
        return None

    def get_output_min_max_from_activation(self, activation_node):
        check_or_raise(
            activation_node.target in self.FUSEABLE_ACTIVATIONS,
            f"Attempted to fuse activation: {activation_node.target}, but it is not a fuseable activation",
        )
        if activation_node.target == exir_ops.edge.aten.relu.default:
            output_min = 0
            output_max = "+inf"
        elif activation_node.target == exir_ops.edge.aten.hardtanh.default:
            output_min = -1
            output_max = 1
            if len(activation_node.args) > 1:
                output_min = activation_node.args[1]
                output_max = activation_node.args[2]

        return OutputMinMax(output_min, output_max)

    def call(self, graph_module: torch.fx.GraphModule):
        for activation_node in graph_module.graph.nodes:
            if activation_node.op == "call_function":
                if activation_node.target in self.FUSEABLE_ACTIVATIONS:
                    preceding_op = activation_node.args[0]
                    if (
                        preceding_op.op == "call_function"
                        and preceding_op.target in self.FUSEABLE_OPS
                    ):
                        # Delete activation, and embed metadata into preceding op
                        output_min_max = self.get_output_min_max_from_activation(
                            activation_node
                        )
                        preceding_op.meta[self.FUSED_ACTIVATION_TAG] = output_min_max
                        activation_node.replace_all_uses_with(preceding_op)
                        graph_module.graph.erase_node(activation_node)

        graph_module.recompile()
        # To Regenerate meta data and shape information, retrace module
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
