# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class RemoveRedundancy(ExportPass):
    """
    Trim certain operators to reduce unnecessary overhead.
    """

    def __init__(self):
        super(RemoveRedundancy, self).__init__()
        self.redundant_ops = {
            torch.clone: self._default_condition,
            torch.ops.aten.clone.default: self._default_condition,
            exir_ops.edge.aten.clone.default: self._default_condition,
            torch.ops.aten.alias.default: self._default_condition,
            exir_ops.edge.aten.alias.default: self._default_condition,
            exir_ops.edge.aten.lift_fresh_copy.default: self._default_condition,
            # remove this target if '_skip_dim_order' is set to False
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default: self._dim_order_op_condition,
            # remove channel_last / contiguous _to_copy if '_skip_dim_order' is set to True
            exir_ops.edge.aten._to_copy.default: self._to_copy_op_condition,
        }

    def _dim_order_op_condition(self, node):
        dim_order = node.kwargs.get("dim_order")
        # skip if there contains layout hint
        # e.g. (0, 2, 3, 1) != (0, 1, 2, 3)
        return dim_order != list(range(len(dim_order)))

    def _to_copy_op_condition(self, node):
        return "memory_format" in node.kwargs

    def _default_condition(self, ndoe):
        return True

    def _remove(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if n.target not in self.redundant_ops or not self.redundant_ops[n.target](
                n
            ):
                continue

            to_be_remove = n
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            graph_module.graph.erase_node(to_be_remove)

    def call(self, graph_module: torch.fx.GraphModule):
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
