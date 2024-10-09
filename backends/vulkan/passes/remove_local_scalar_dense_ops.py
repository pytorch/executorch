# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def remove_local_scalar_dense_ops(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    Remove local_scalar_dense op nodes and replace uses with parent node, or the
    original scalar tensor.
    """
    target_op = torch.ops.aten._local_scalar_dense.default
    for node in graph.nodes:
        if node.op == "call_function" and node.target == target_op:
            replace_node = node.args[0]
            # If the argument to the local_scalar_dense op is a select op with only
            # one user, and the argument to the select op is a tensor with only one
            # element (i.e. a scalar tensor), then replace the entire pattern with the
            # scalar tensor.
            if (
                replace_node.op == "call_function"
                and replace_node.target == exir_ops.edge.aten.select_copy.int
            ):
                if replace_node.args[0].meta["val"].numel() == 1:
                    replace_node = replace_node.args[0]

            with graph.inserting_after(node):
                node.replace_all_uses_with(replace_node)

    graph.eliminate_dead_code()
    return graph


class RemoveLocalScalarDenseOpsTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = remove_local_scalar_dense_ops(graph_module.graph)
        return PassResult(graph_module, True)
