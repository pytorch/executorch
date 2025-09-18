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

class FuseConvBinaryOpPass(ExportPass):
    """
    Some activations like ReLU and hardtanh can be fused with certain operators (e.g. convolution) preceding it.
    """

    FUSEABLE_OPS = [
        exir_ops.edge.aten.convolution.default,
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
        

    def call(self, graph_module: torch.fx.GraphModule):
        
        fuseAdded = True
        while fuseAdded:
            fuseAdded = False
            for arg_idx in range(0, 2):
                for binary_op_node in graph_module.graph.nodes:
                    if binary_op_node.op == "call_function":
                        if binary_op_node.target in self.FUSEABLE_BINARY_OPS:
                            preceding_op = binary_op_node.args[arg_idx]
                            if (
                                preceding_op.op == "call_function"
                                and preceding_op.target in self.FUSEABLE_OPS
                            ):

                                # For now only pw conv2d s1p0 is supported
                                if not (len(preceding_op.args[3]) == 2 and preceding_op.args[3][0] == 1 and preceding_op.args[3][1] == 1 and preceding_op.args[4][0] == 0 and preceding_op.args[4][1] == 0):
                                    continue

                                # Ensure the shapes match
                                if "val" not in binary_op_node.args[0].meta or "val" not in binary_op_node.args[1].meta:
                                    continue
                                if len(binary_op_node.args[0].meta["val"].shape) != len(binary_op_node.args[1].meta["val"].shape):
                                    continue
                                

                                # Get the texture to do the binary op
                                texture = binary_op_node.args[(arg_idx + 1)%2]

                                # Fuse only if the texture exists before the preceding op
                                if not self.exists_before(graph_module, texture, preceding_op):
                                    continue

                                new_args = list(preceding_op.args)
                                new_args.append(texture)
                                new_args = tuple(new_args)
                                binary_op_node.replace_all_uses_with(preceding_op)
                                graph_module.graph.erase_node(binary_op_node)

                                new_op = None
                                if binary_op_node.target == exir_ops.edge.aten.add.Tensor:
                                    new_op = exir_ops.edge.et_vk.conv_with_binary_add.default
                                if binary_op_node.target == exir_ops.edge.aten.sub.Tensor:
                                    new_op = exir_ops.edge.et_vk.conv_with_binary_sub.default
                                if binary_op_node.target == exir_ops.edge.aten.mul.Tensor:
                                    new_op = exir_ops.edge.et_vk.conv_with_binary_mul.default
                                if binary_op_node.target == exir_ops.edge.aten.div.Tensor:
                                    new_op = exir_ops.edge.et_vk.conv_with_binary_div.default

                                assert(new_op != None)

                                # Create and insert node of custom op `conv_with_binary_op`
                                with graph_module.graph.inserting_before(preceding_op):
                                    conv_binary_op_node = graph_module.graph.create_node(
                                        "call_function",
                                        new_op,
                                        new_args,
                                    )

                                    preceding_op.replace_all_uses_with(conv_binary_op_node)
                                    graph_module.graph.erase_node(preceding_op)
                                
                                fuseAdded = True

                graph_module.recompile()
                graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
