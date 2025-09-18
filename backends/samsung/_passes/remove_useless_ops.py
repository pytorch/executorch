# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx import GraphModule


class RemoveUselessOpPass(ExportPass):
    # such ops should be single-in and single-out
    USELESS_OP_SET = {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.alias.default,
        exir_ops.edge.aten.lift_fresh_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }

    def __init__(self):
        super().__init__()

    def _remove_useless(
        self,
        graph_module: GraphModule,
    ):
        for node in graph_module.graph.nodes:
            if node.target not in self.USELESS_OP_SET:
                continue

            # Prevent from removing if data type may change.
            if (
                node.target == exir_ops.edge.aten._to_copy.default
                or node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            ) and "memory_format" not in node.kwargs:
                continue

            for user in [user for user in node.users.keys()]:  # noqa: C416
                user.replace_input_with(node, node.all_input_nodes[0])
            graph_module.graph.erase_node(node)

    def call(self, graph_module: GraphModule):
        self._remove_useless(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
