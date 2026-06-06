# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

from .utils import copy_meta, get_const_node


class DecomposeRemainder(ExportPass):
    """
    Decompose remainder.Scalar and remainder.Tensor using the identity:
        remainder(x, y) = x - floor(x / y) * y
    """

    def __init__(self):
        super(DecomposeRemainder, self).__init__()
        self.remainder_targets = {
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
            exir_ops.edge.aten.remainder.Scalar,
            exir_ops.edge.aten.remainder.Tensor,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # Cache scalar:node mappings to avoid duplicate buffer registrations if the same scalar divisor appears in multiple remainder ops
        const_cache = {}

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.remainder_targets:
                x_node = node.args[0]
                y_arg = node.args[1]
                is_edge = isinstance(node.target, EdgeOpOverload)
                meta = node.meta

                div_op = (
                    exir_ops.edge.aten.div.Tensor
                    if is_edge
                    else torch.ops.aten.div.Tensor
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
                sub_op = (
                    exir_ops.edge.aten.sub.Tensor
                    if is_edge
                    else torch.ops.aten.sub.Tensor
                )

                is_scalar = not isinstance(y_arg, torch.fx.Node)
                if is_scalar and is_edge:
                    if y_arg not in const_cache:
                        attr_name = get_new_attr_name_with_prefix("_remainder_const_")(
                            graph_module
                        )
                        const_cache[y_arg] = get_const_node(
                            graph, graph_module, attr_name, y_arg, node
                        )
                    y_node = const_cache[y_arg]
                else:
                    y_node = y_arg

                with graph.inserting_before(node):
                    div_node = graph.create_node(
                        "call_function", div_op, (x_node, y_node)
                    )
                    div_node.meta = copy_meta(meta)

                    floor_node = graph.create_node(
                        "call_function", floor_op, (div_node,)
                    )
                    floor_node.meta = copy_meta(meta)

                    mul_node = graph.create_node(
                        "call_function", mul_op, (floor_node, y_node)
                    )
                    mul_node.meta = copy_meta(meta)

                    sub_node = graph.create_node(
                        "call_function", sub_op, (x_node, mul_node)
                    )
                    sub_node.meta = copy_meta(meta)

                for user in node.users.copy():
                    user.replace_input_with(node, sub_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
