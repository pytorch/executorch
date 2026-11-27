# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta


class DecomposeFill(ExportPass):
    """
    Decompose fill.Scalar into full.default.
    fill(input, value) is semantically equivalent to full(input.shape, value).
    """

    def __init__(self):
        super().__init__()
        self.targets = {
            torch.ops.aten.fill.Scalar,
            torch.ops.aten.fill_.Scalar,
            exir_ops.edge.aten.fill.Scalar,
            exir_ops.edge.aten.fill_.Scalar,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.targets:
                fill_node = node
                is_edge = isinstance(node.target, EdgeOpOverload)
                input_node = node.args[0]
                scalar_value = node.args[1]

                # Get the shape from the input tensor metadata
                shape = list(input_node.meta["val"].shape)

                full_op = (
                    exir_ops.edge.aten.full.default
                    if is_edge
                    else torch.ops.aten.full.default
                )

                with graph.inserting_after(input_node):
                    full_node = graph.create_node(
                        "call_function",
                        full_op,
                        (shape, scalar_value),
                    )
                    full_node.meta = copy_meta(fill_node.meta)

                for user in fill_node.users.copy():
                    user.replace_input_with(fill_node, full_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
