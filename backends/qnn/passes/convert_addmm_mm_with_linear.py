# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertAddmmmmWithLinear(ExportPass):
    def __init__(self):
        super(ConvertAddmmmmWithLinear, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        ops = exir_ops.edge
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and (
                node.target == ops.aten.mm.default
                or node.target == ops.aten.addmm.default
            ):
                with graph.inserting_after(node):
                    if node.target == ops.aten.addmm.default:
                        weight_t_node = node.args[2]
                        if weight_t_node.target != ops.aten.permute_copy.default:
                            continue
                        weight_node = weight_t_node.args[0]
                        args = (node.args[1], weight_node, node.args[0])
                        linear_node = graph.create_node(
                            "call_function", ops.aten.linear.default, args
                        )
                        node.replace_all_uses_with(linear_node)
                        output_val = linear_node.target(
                            args[0].meta["val"],
                            args[1].meta["val"],
                            args[2].meta["val"],
                        )
                    else:
                        weight_t_node = node.args[1]
                        if weight_t_node.target != ops.aten.permute_copy.default:
                            continue
                        weight_node = weight_t_node.args[0]
                        args = (node.args[0], weight_node)
                        linear_node = graph.create_node(
                            "call_function", ops.aten.linear.default, args
                        )
                        node.replace_all_uses_with(linear_node)
                        output_val = linear_node.target(
                            args[0].meta["val"], args[1].meta["val"]
                        )
                    linear_node.meta = node.meta
                    linear_node.meta["val"] = output_val

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
