# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeTan(ExportPass):
    """
    Decompose tan(x) = sin(x) / cos(x)
    """

    def __init__(self):
        super(DecomposeTan, self).__init__()
        self.targets = {
            torch.ops.aten.tan.default,
            exir_ops.edge.aten.tan.default,
        }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.targets:
                is_edge = isinstance(node.target, EdgeOpOverload)

                sin_op = (
                    exir_ops.edge.aten.sin.default
                    if is_edge
                    else torch.ops.aten.sin.default
                )
                cos_op = (
                    exir_ops.edge.aten.cos.default
                    if is_edge
                    else torch.ops.aten.cos.default
                )
                div_op = (
                    exir_ops.edge.aten.div.Tensor
                    if is_edge
                    else torch.ops.aten.div.Tensor
                )

                with graph.inserting_before(node):
                    sin_node = graph.create_node(
                        "call_function", sin_op, (node.args[0],)
                    )
                    sin_node.meta = copy_meta(node.meta)

                    cos_node = graph.create_node(
                        "call_function", cos_op, (node.args[0],)
                    )
                    cos_node.meta = copy_meta(node.meta)

                    div_node = graph.create_node(
                        "call_function", div_op, (sin_node, cos_node)
                    )
                    div_node.meta = copy_meta(node.meta)

                for user in node.users.copy():
                    user.replace_input_with(node, div_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
