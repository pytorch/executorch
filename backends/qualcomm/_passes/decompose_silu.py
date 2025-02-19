# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeSilu(ExportPass):
    def __init__(self):
        super(DecomposeSilu, self).__init__()

    def _copy_meta(self, meta: Dict):
        copied = {}
        for k, v in meta.items():
            copied[k] = v
        return copied

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.silu.default
            ):
                silu_node = node
                silu_node_input = node.args[0]
                with graph_module.graph.inserting_after(silu_node_input):
                    sigmoid_node = graph.create_node(
                        "call_function", torch.ops.aten.sigmoid, (silu_node_input,)
                    )
                    sigmoid_node.meta = self._copy_meta(silu_node.meta)
                    with graph_module.graph.inserting_after(sigmoid_node):
                        mul_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.mul,
                            (silu_node_input, sigmoid_node),
                        )
                        mul_node.meta = self._copy_meta(silu_node.meta)
                        for user in silu_node.users.copy():
                            user.replace_input_with(silu_node, mul_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
