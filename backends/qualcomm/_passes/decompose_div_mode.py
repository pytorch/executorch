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


class DecomposeDivMode(ExportPass):
    """
    Decompose aten.div.Tensor_mode into supported primitives.

    div(x, y, rounding_mode=None)    -> div(x, y)
    div(x, y, rounding_mode="trunc") -> trunc(div(x, y))
    div(x, y, rounding_mode="floor") -> floor(div(x, y))

    Note: div.Scalar_mode is handled by LiftConstantScalarOperands which converts it to div.Tensor_mode before this pass runs.
    """

    def __init__(self):
        super(DecomposeDivMode, self).__init__()
        self.targets = {
            torch.ops.aten.div.Tensor_mode,
            exir_ops.edge.aten.div.Tensor_mode,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.targets:
                is_edge = isinstance(node.target, EdgeOpOverload)
                meta = node.meta

                x_node = node.args[0]
                y_node = node.args[1]

                rounding_mode = node.kwargs.get("rounding_mode", None)
                if rounding_mode is None and len(node.args) > 2:
                    rounding_mode = node.args[2]

                div_op = (
                    exir_ops.edge.aten.div.Tensor
                    if is_edge
                    else torch.ops.aten.div.Tensor
                )

                with graph.inserting_before(node):
                    # Step 1: div_result = div(x, y)
                    div_node = graph.create_node(
                        "call_function", div_op, (x_node, y_node)
                    )
                    div_node.meta = copy_meta(meta)

                    # Step 2: Apply rounding mode if needed
                    if rounding_mode == "trunc":
                        trunc_op = (
                            exir_ops.edge.aten.trunc.default
                            if is_edge
                            else torch.ops.aten.trunc.default
                        )
                        result_node = graph.create_node(
                            "call_function", trunc_op, (div_node,)
                        )
                        result_node.meta = copy_meta(meta)
                    elif rounding_mode == "floor":
                        floor_op = (
                            exir_ops.edge.aten.floor.default
                            if is_edge
                            else torch.ops.aten.floor.default
                        )
                        result_node = graph.create_node(
                            "call_function", floor_op, (div_node,)
                        )
                        result_node.meta = copy_meta(meta)
                    else:
                        # rounding_mode=None: plain division
                        result_node = div_node

                for user in node.users.copy():
                    user.replace_input_with(node, result_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
