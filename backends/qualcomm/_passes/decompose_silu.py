# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeSilu(ExportPass):
    def __init__(self):
        super(DecomposeSilu, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and node.target in {
                torch.ops.aten.silu.default,
                torch.ops.aten.silu_.default,
            }:
                silu_node = node
                silu_node_input = node.args[0]
                with graph_module.graph.inserting_after(silu_node_input):
                    sigmoid_node = graph.create_node(
                        "call_function",
                        torch.ops.aten.sigmoid.default,
                        (silu_node_input,),
                    )
                    sigmoid_node.meta = copy_meta(silu_node.meta)
                    with graph_module.graph.inserting_after(sigmoid_node):
                        mul_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.mul.Tensor,
                            (silu_node_input, sigmoid_node),
                        )
                        mul_node.meta = copy_meta(silu_node.meta)
                        for user in silu_node.users.copy():
                            user.replace_input_with(silu_node, mul_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
