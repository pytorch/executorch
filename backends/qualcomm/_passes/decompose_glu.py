# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


# this wrapper is required for IO name mapping with decomposed graph
class Glu(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.glu = torch.nn.GLU(dim=dim)

    def forward(self, x):
        return self.glu(x)


class DecomposeGlu(ExportPass):
    """
    Decompose glu for quantization annotation to work properly.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.glu.default:
                ep = torch.export.export(
                    Glu(dim=-1 if len(node.args) < 2 else node.args[1]),
                    (node.args[0].meta["val"],),
                )
                decomposed_module = ep.run_decompositions().graph_module

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": node.args[0]}
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
