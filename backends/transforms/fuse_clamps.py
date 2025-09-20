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

class FuseClampsPass(ExportPass):

    FUSEABLE_CLAMPS = [
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.clamp.default,
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
        elif activation_node.target == exir_ops.edge.aten.clamp.default:
            output_min = None
            output_max = None
            if len(activation_node.args) >= 2:
                output_min = activation_node.args[1]
            if len(activation_node.args) >= 3:
                output_max = activation_node.args[2]

        return output_min, output_max
        

    def call(self, graph_module: torch.fx.GraphModule):
        fuseAdded = True
        while fuseAdded:
            fuseAdded = False
            for clamp_2_node in graph_module.graph.nodes:
                if clamp_2_node.op == "call_function":
                    if clamp_2_node.target in self.FUSEABLE_CLAMPS:
                        preceding_op = clamp_2_node.args[0]
                        if (
                            preceding_op.op == "call_function"
                            and preceding_op.target in self.FUSEABLE_CLAMPS
                        ):
                            # Ensure the shapes match
                            if "val" not in clamp_2_node.args[0].meta or "val" not in preceding_op.args[0].meta:
                                continue
                            if len(clamp_2_node.args[0].meta["val"].shape) != len(preceding_op.args[0].meta["val"].shape):
                                continue

                            min_max1 = self.get_output_min_max_from_activation(preceding_op)
                            min_max2 = self.get_output_min_max_from_activation(clamp_2_node)

                            min_max = [None, None]

                            if min_max1[0] is None and min_max2[0] is not None:
                                min_max[0] = min_max2[0]
                            elif min_max1[0] is not None and min_max2[0] is None:
                                min_max[0] = min_max1[0]
                            else:
                                min_max[0] = min(min_max1[0], min_max2[0])
                            
                            if min_max1[1] is None and min_max2[1] is not None:
                                min_max[1] = min_max2[1]
                            elif min_max1[1] is not None and min_max2[1] is None:
                                min_max[1] = min_max1[1]
                            else:
                                min_max[1] = max(min_max1[1], min_max2[1])

                            new_args = list(preceding_op.args)

                            # Insert the new min/max at indices 1 and 2
                            new_args.insert(1, min_max[0])
                            new_args.insert(2, min_max[1])
                            new_args = new_args[0:3]
                            preceding_op.args = tuple(new_args)
                            clamp_2_node.replace_all_uses_with(preceding_op)
                            graph_module.graph.erase_node(clamp_2_node)
                            fuseAdded = True

            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
