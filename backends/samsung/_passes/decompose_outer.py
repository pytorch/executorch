# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class Outer(torch.nn.Module):
    """outer(a, b) == a.reshape(-1, 1) * b.reshape(1, -1) for 1-D inputs."""

    def forward(self, a, b):
        return a.reshape(-1, 1) * b.reshape(1, -1)


class DecomposeOuter(ExportPass):
    """
    Decompose torch.ops.aten.outer.default into view and mul.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target != torch.ops.aten.outer.default:
                continue
            input_a, input_b = node.args[0], node.args[1]
            decomposed_module = torch.export.export(
                Outer(), (input_a.meta["val"], input_b.meta["val"]), strict=True
            ).module()

            with graph.inserting_before(node):
                remap = {"a": input_a, "b": input_b}
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
