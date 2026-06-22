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


class DecomposeTrunc(ExportPass):
    """
    Decompose trunc via the identity: trunc(x) = sign(x) * floor(abs(x)).
    """

    def __init__(self):
        super(DecomposeTrunc, self).__init__()
        self.trunc_targets = {
            torch.ops.aten.trunc.default,
            exir_ops.edge.aten.trunc.default,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and node.target in self.trunc_targets:
                trunc_node = node
                input_node = node.args[0]

                is_edge = isinstance(node.target, EdgeOpOverload)
                sign_op = (
                    exir_ops.edge.aten.sign.default
                    if is_edge
                    else torch.ops.aten.sign.default
                )
                abs_op = (
                    exir_ops.edge.aten.abs.default
                    if is_edge
                    else torch.ops.aten.abs.default
                )
                floor_op = (
                    exir_ops.edge.aten.floor.default
                    if is_edge
                    else torch.ops.aten.floor.default
                )
                mul_op = (
                    exir_ops.edge.aten.mul.Tensor
                    if is_edge
                    else torch.ops.aten.mul.Tensor
                )

                with graph_module.graph.inserting_after(input_node):
                    sign_node = graph.create_node(
                        "call_function",
                        sign_op,
                        (input_node,),
                    )
                    sign_node.meta = copy_meta(trunc_node.meta)

                    with graph_module.graph.inserting_after(sign_node):
                        abs_node = graph.create_node(
                            "call_function",
                            abs_op,
                            (input_node,),
                        )
                        abs_node.meta = copy_meta(trunc_node.meta)

                        with graph_module.graph.inserting_after(abs_node):
                            floor_node = graph.create_node(
                                "call_function",
                                floor_op,
                                (abs_node,),
                            )
                            floor_node.meta = copy_meta(trunc_node.meta)

                            with graph_module.graph.inserting_after(floor_node):
                                mul_node = graph.create_node(
                                    "call_function",
                                    mul_op,
                                    (sign_node, floor_node),
                                )
                                mul_node.meta = copy_meta(trunc_node.meta)

                                for user in trunc_node.users.copy():
                                    user.replace_input_with(trunc_node, mul_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
