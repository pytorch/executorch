# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.experimental.proxy_tensor import make_fx

from .utils import merge_decomposed_graph


class DecomposeEinsum(ExportPass):
    """
    Decompose einsum for quantization annotation to work properly.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.einsum.default:
                decomposed_module = make_fx(
                    node.target,
                    tracing_mode="fake",
                )(node.args[0], [arg.meta["val"] for arg in node.args[1]])

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {}
                    # Different from other nodes, einsum args[0] is the einsum equation,
                    # while input nodes are stored in args[1]
                    for i, arg in enumerate(node.args[1]):
                        remap[f"arg1_{i+1}"] = arg

                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                        predicate=lambda decomp_node: "arg0" not in decomp_node.name,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
