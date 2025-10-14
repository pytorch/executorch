# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class DecomposeModule(torch.nn.Module):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x):
        return torch.where(x <= self.threshold, self.value, x)


class DecomposeThreshold(ExportPass):
    """
    Decompose threshold to less_equal and where.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target in {
                torch.ops.aten.threshold_.default,
                torch.ops.aten.threshold.default,
            }:
                input_node = node.args[0]
                threshold = node.args[1]
                value = node.args[2]

                model = DecomposeModule(threshold, value)
                decomposed_module = torch.export.export(
                    model, (input_node.meta["val"],), strict=True
                ).module()

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": input_node}
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
