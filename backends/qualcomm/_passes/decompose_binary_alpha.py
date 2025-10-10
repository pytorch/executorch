# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta

decomp_set = {torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor}


class DecomposeBinaryAlpha(ExportPass):
    """
    QNN does not support alpha parameter for add/sub.
    Decompose to mul + add / mul + sub
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                node.target in decomp_set
                and "alpha" in node.kwargs
                and node.kwargs["alpha"] != 1
            ):
                alpha = node.kwargs["alpha"]
                # Remove alpha from immutable dict
                node.kwargs = {k: v for k, v in node.kwargs.items() if k != "alpha"}
                input2_node = node.args[1]
                # If input2 is constant, we can just multiply the value for optimization
                if isinstance(input2_node, (int, float)):
                    arg_list = list(node.args)
                    arg_list[1] = input2_node * alpha
                    node.args = tuple(arg_list)
                    continue
                with graph.inserting_before(node):
                    mul_op = torch.ops.aten.mul.Scalar
                    mul_node = graph.create_node(
                        "call_function",
                        mul_op,
                        (
                            input2_node,
                            alpha,
                        ),
                    )
                    mul_node.meta = copy_meta(node.meta)
                    node.replace_input_with(input2_node, mul_node)
                    node.args = (
                        node.args[0],
                        mul_node,
                    )

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
