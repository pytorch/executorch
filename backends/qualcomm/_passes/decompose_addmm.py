# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta, create_const_node


class DecomposeAddmm(ExportPass):
    """
    Decompose addmm into mm + add (with optional mul for non-unit alpha/beta).
        addmm(bias, input, mat2, beta=1, alpha=1) = beta * bias + alpha * (input @ mat2)

    For the common case (alpha=1, beta=1): addmm(bias, input, mat2) = mm(input, mat2) + bias

    Note: This pass serves as a fallback for standalone addmm nodes that are NOT
    handled by the ExecuTorch-provided pass AddmmToLinearTransform.
    Any remaining addmm nodes (e.g., with non-transposed mat2) are decomposed here into mm + add.
    """

    def __init__(self):
        super().__init__()
        self.addmm_targets = {
            torch.ops.aten.addmm.default,
            exir_ops.edge.aten.addmm.default,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.addmm_targets:
                is_edge = isinstance(node.target, EdgeOpOverload)
                bias_node = node.args[0]
                input_node = node.args[1]
                mat2_node = node.args[2]
                # kwargs beta and alpha default to 1
                beta = node.kwargs.get("beta", 1)
                alpha = node.kwargs.get("alpha", 1)

                mm_op = (
                    exir_ops.edge.aten.mm.default
                    if is_edge
                    else torch.ops.aten.mm.default
                )
                add_op = (
                    exir_ops.edge.aten.add.Tensor
                    if is_edge
                    else torch.ops.aten.add.Tensor
                )
                mul_op = (
                    exir_ops.edge.aten.mul.Tensor
                    if is_edge
                    else torch.ops.aten.mul.Tensor
                )

                meta = node.meta

                with graph.inserting_before(node):
                    # mm_result = input @ mat2
                    mm_node = graph.create_node(
                        "call_function", mm_op, (input_node, mat2_node)
                    )
                    mm_node.meta = copy_meta(meta)

                    if alpha != 1:
                        alpha_node = create_const_node(
                            graph,
                            graph_module,
                            f"{node.name}_alpha",
                            alpha,
                            mm_node,
                        )
                        mm_scaled = graph.create_node(
                            "call_function", mul_op, (mm_node, alpha_node)
                        )
                        mm_scaled.meta = copy_meta(meta)
                        mm_result = mm_scaled
                    else:
                        mm_result = mm_node

                    if beta != 1:
                        beta_const = create_const_node(
                            graph,
                            graph_module,
                            f"{node.name}_beta",
                            beta,
                            bias_node,
                        )
                        bias_scaled = graph.create_node(
                            "call_function", mul_op, (bias_node, beta_const)
                        )
                        bias_scaled.meta = copy_meta(meta)
                        bias_result = bias_scaled
                    else:
                        bias_result = bias_node

                    # result = mm_result + bias
                    add_node = graph.create_node(
                        "call_function", add_op, (mm_result, bias_result)
                    )
                    add_node.meta = copy_meta(meta)

                for user in node.users.copy():
                    user.replace_input_with(node, add_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
