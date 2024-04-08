# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FuseConsecutiveTranspose(ExportPass):
    """
    This pass fuses consecutive transpose / permute into one to reduce runtime
    overhead
    """

    def __init__(self):
        super().__init__()
        self.op_map = {
            exir_ops.edge.aten.permute_copy.default,
        }
        self.visited = set()
        self.nodes = []

    def _traverse(self, node):
        if node.op == "call_function" and node.target in self.op_map:
            self.nodes.append(node)
            self.visited.add(node)
            if len(node.users) == 1:
                self._traverse(list(node.users)[0])

    def _fuse(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph = graph_module.graph
        for n in graph_module.graph.nodes:
            if n in self.visited:
                continue
            if n.op == "call_function" and n.target in self.op_map:
                self._traverse(n)
            num_nodes = len(self.nodes)
            if num_nodes > 1:
                input_node, output_node = self.nodes[0].args[0], self.nodes[-1]
                input_shape = input_node.meta["val"].shape
                axis_order = torch.arange(len(input_shape)).tolist()
                for node in self.nodes:
                    axis_order = [axis_order[i] for i in node.args[1]]
                with graph.inserting_after(input_node):
                    permute_op = exir_ops.edge.aten.permute_copy.default
                    permute_node = graph.create_node(
                        "call_function", permute_op, (input_node, axis_order)
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, permute_node)
                    # copy metadata

                    permute_node.meta = output_node.meta
            # clear current stack

            self.nodes = []

    def call(self, graph_module: torch.fx.GraphModule):
        self._fuse(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
