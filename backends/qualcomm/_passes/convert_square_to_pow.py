# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class ConvertSquareToPow(ExportPass):
    """
    Convert square to pow with a scalar value of 2.
    This allows LiftConstantScalarOperands to lift the scalar into a scalar.
    Otherwise, the square op will be converted to pow.tensor_scalar after to_edge.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.square.default:
                input_node = node.args[0]
                with graph_module.graph.inserting_after(input_node):
                    pow_op = torch.ops.aten.pow.Tensor_Scalar
                    pow_node = graph.create_node(
                        "call_function", pow_op, (input_node, 2)
                    )
                    pow_node.meta = copy_meta(node.meta)
                for user in node.users.copy():
                    user.replace_input_with(node, pow_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
