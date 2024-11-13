# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class InsertSqueezeAfterSumPass(ExportPass):
    """
    In Pytorch, the default behaviour of Tensor.sum is to squeeze
    the dimension that is summed (keep_dim = False).
    However, in TOSA, REDUCE_SUM always preserves the
    rank of the input (keep_dim = True).
    To get a 1-1 mapping in the sum lowering, normalize the
    keep_dim = False case to keep_dim = True and add squeeze ops.

    Original:
        sum(dims, keep_dim = False)
    After pass:
        sum(dims, keep_dim = True)
        squeeze(dim = dims)
    """

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.sum.dim_IntList:
                continue
            sum_node = cast(torch.fx.Node, node)
            keep_dim = cast(bool, sum_node.args[2] if len(sum_node.args) > 2 else False)
            if keep_dim:
                continue

            dim_list = cast(list[int], sum_node.args[1])

            # Add keep_dim = True arg to sum node.
            sum_node.args = sum_node.args[0:2] + (True,)

            with graph_module.graph.inserting_after(sum_node):
                squeeze_node = create_node(
                    graph_module.graph, exir_ops.edge.aten.squeeze_copy.dims, ()
                )
                sum_node.replace_all_uses_with(squeeze_node)
                squeeze_node.args = (sum_node, dim_list)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
