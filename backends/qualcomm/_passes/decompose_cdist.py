# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class CDist(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Step 1: Compute differences
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)

        # Step 2: Square differences
        sq_diff = diff**2

        # Step 3: Sum of squares
        sum_sq_diff = sq_diff.sum(dim=-1)

        # Step 4: Square root
        distances = torch.sqrt(sum_sq_diff)

        return distances


class DecomposeCDist(ExportPass):
    """
    Decompose for math equivalent op.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            model = CDist()
            if torch.ops.aten.cdist.default == node.target:
                if len(node.args) > 2:
                    assert (
                        node.args[2] == 2
                    ), "Currently only p=2 is supported for CDist Decomposition"
                decomposed_module = torch.export.export(
                    model,
                    (node.args[0].meta["val"], node.args[1].meta["val"]),
                    strict=True,
                ).module()
                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": node.args[0], "y": node.args[1]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
