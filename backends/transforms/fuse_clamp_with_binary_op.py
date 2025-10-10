# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import executorch.backends.vulkan.custom_ops_lib  # noqa

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class FuseClampBinaryOpPass(ExportPass):

    FUSEABLE_CLAMP_OPS = [
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.clamp.default,
    ]
    FUSEABLE_BINARY_OPS = [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
    ]

    def exists_before(self, graph_module, node_a, node_b):
        seen_a = False
        for n in graph_module.graph.nodes:
            if n is node_a:
                seen_a = True
            if n is node_b:
                return seen_a
        return False

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
        elif activation_node.target == exir_ops.edge.aten.clamp.default:
            output_min = None
            output_max = None
            if len(activation_node.args) >= 2:
                output_min = activation_node.args[1]
            if len(activation_node.args) >= 3:
                output_max = activation_node.args[2]

        return output_min, output_max

    def fuse_binary_op_with_clamp(self, graph_module: torch.fx.GraphModule):
        fuseAdded = False
        for clamp_node in graph_module.graph.nodes:
            if clamp_node.op == "call_function":
                if clamp_node.target in self.FUSEABLE_CLAMP_OPS:
                    preceding_op = clamp_node.args[0]

                    if (
                        preceding_op.op == "call_function"
                        and preceding_op.target in self.FUSEABLE_BINARY_OPS
                    ):
                        # Delete activation
                        output_min_max = self.get_output_min_max_from_activation(
                            clamp_node
                        )
                        new_args = list(preceding_op.args)
                        new_args.append(output_min_max[0])
                        new_args.append(output_min_max[1])
                        new_args = tuple(new_args)
                        clamp_node.replace_all_uses_with(preceding_op)
                        graph_module.graph.erase_node(clamp_node)

                        new_op = None
                        match preceding_op.target:
                            case exir_ops.edge.aten.add.Tensor:
                                new_op = (
                                    exir_ops.edge.et_vk.binary_add_with_clamp.default
                                )
                            case exir_ops.edge.aten.sub.Tensor:
                                new_op = (
                                    exir_ops.edge.et_vk.binary_sub_with_clamp.default
                                )
                            case exir_ops.edge.aten.mul.Tensor:
                                new_op = (
                                    exir_ops.edge.et_vk.binary_mul_with_clamp.default
                                )
                            case exir_ops.edge.aten.div.Tensor:
                                new_op = (
                                    exir_ops.edge.et_vk.binary_div_with_clamp.default
                                )

                        # Create and insert node of custom op `binary_<op>_with_clamp`
                        with graph_module.graph.inserting_before(preceding_op):
                            binary_op_clamp_node = graph_module.graph.create_node(
                                "call_function",
                                new_op,
                                new_args,
                            )

                            preceding_op.replace_all_uses_with(binary_op_clamp_node)
                            graph_module.graph.erase_node(preceding_op)

                            fuseAdded = True

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return [fuseAdded, graph_module]

    def call(self, graph_module: torch.fx.GraphModule):
        fuseAdded = True
        while fuseAdded:
            fuseAdded, graph_module = self.fuse_binary_op_with_clamp(graph_module)

        return PassResult(graph_module, True)
