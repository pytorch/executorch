# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeExpM1(ExportPass):
    """
    Decompose for expm1 to exponential and minus 1.
    """

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.special_expm1.default:
                input_node = node.args[0]
                with graph_module.graph.inserting_after(input_node):
                    exp_op = torch.ops.aten.exp.default
                    exp_node = graph.create_node("call_function", exp_op, (input_node,))
                    exp_node.meta = copy_meta(node.meta)
                    with graph_module.graph.inserting_after(exp_node):
                        sub_op = torch.ops.aten.sub.Tensor
                        sub_node = graph.create_node(
                            "call_function",
                            sub_op,
                            (
                                exp_node,
                                1,
                            ),
                        )
                        sub_node.meta = copy_meta(node.meta)
                for user in node.users.copy():
                    user.replace_input_with(node, sub_node)
                graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
