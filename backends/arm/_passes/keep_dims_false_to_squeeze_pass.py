# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_node_arg,
    set_node_arg,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class KeepDimsFalseToSqueezePass(ExportPass):
    """
    In Pytorch, the default behaviour of for example Tensor.sum is to squeeze
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

    # CURRENTLY NOT HANDLED OPS
    # exir_ops.edge.aten.amax,
    # exir_ops.edge.aten.amin,
    # exir_ops.edge.aten.any.dim,
    # exir_ops.edge.aten.any.dims,
    # exir_ops.edge.aten.argmax,
    # exir_ops.edge.aten.argmin,
    # exir_ops.edge.aten.max.dim,
    # exir_ops.edge.aten.min.dim,
    # exir_ops.edge.aten.prod.dim_int,

    # HANDLED OPS
    # exir_ops.edge.aten.sum.dim_IntList
    # exir_ops.edge.aten.var.correction (decomposed in decompose_var_pass)
    # exir_ops.edge.aten.var.dim (decomposed in decompose_var_pass)
    # exir_ops.edge.aten.mean.dim (decomposed in decompose_meandim_pass)

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            keep_dim_index = None

            if node.op != "call_function":
                continue
            if node.target == exir_ops.edge.aten.sum.dim_IntList:
                keep_dim_index = 2
            else:
                continue

            sum_node = cast(torch.fx.Node, node)
            keep_dim = get_node_arg(
                # pyre-ignore[6]
                sum_node.args,
                keep_dim_index,
                False,
            )

            if keep_dim:
                continue

            dim_list = get_node_arg(sum_node.args, 1, [0])  # pyre-ignore[6]

            # Add keep_dim = True arg to sum node.
            set_node_arg(sum_node, 2, True)

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
