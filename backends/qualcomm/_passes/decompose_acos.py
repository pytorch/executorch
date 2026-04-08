# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta, get_const_node


class DecomposeAcos(ExportPass):
    """
    Decompose acos using the identity: acos(x) = π/2 - asin(x).
    """

    def __init__(self):
        super(DecomposeAcos, self).__init__()
        self.acos_targets = {
            torch.ops.aten.acos.default,
            exir_ops.edge.aten.acos.default,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        acos_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target in self.acos_targets
        ]
        if not acos_nodes:
            return PassResult(graph_module, False)

        pi_half = torch.pi / 2.0
        pi_half_node = None

        for node in acos_nodes:
            input_node = node.args[0]
            is_edge = isinstance(node.target, EdgeOpOverload)

            asin_op = (
                exir_ops.edge.aten.asin.default
                if is_edge
                else torch.ops.aten.asin.default
            )
            sub_op = (
                exir_ops.edge.aten.sub.Tensor if is_edge else torch.ops.aten.sub.Tensor
            )

            if is_edge and pi_half_node is None:
                pi_half_node = get_const_node(
                    graph, graph_module, "_pi_half_constant", pi_half, node
                )

            sub_arg = pi_half_node if is_edge else pi_half

            with graph.inserting_before(node):
                asin_node = graph.create_node("call_function", asin_op, (input_node,))
                asin_node.meta = copy_meta(node.meta)

                sub_node = graph.create_node(
                    "call_function", sub_op, (sub_arg, asin_node)
                )
                sub_node.meta = copy_meta(node.meta)

            for user in node.users.copy():
                user.replace_input_with(node, sub_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
