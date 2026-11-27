# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

from .utils import merge_decomposed_graph


class DecomposeHardsigmoid(ExportPass):
    """
    Decompose `aten.hardsigmoid` into mathematically equivalent ops
    by leveraging the decomposition table to Core ATen.
    """

    def _output_processor(
        self, target_node: torch.fx.Node, output_node: torch.fx.Node, remap: Dict
    ):
        for user in target_node.users.copy():
            user.replace_input_with(
                target_node,
                remap[output_node.args[0]],
            )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.hardsigmoid.default:
                decomp_mappings = get_decompositions([node.target])
                decomposed_module = make_fx(
                    node.target,
                    decomposition_table=decomp_mappings,
                    tracing_mode="fake",
                )(node.args[0].meta["val"])

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"arg0_1": node.args[0]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                        output_processor=self._output_processor,
                    )
                    graph.erase_node(node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
