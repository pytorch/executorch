# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

from .utils import merge_decomposed_graph


class DecomposeTriu(ExportPass):
    """
    Decompose triu during quantization or export stage
    This allows LiftConstantScalarOperands to lift the scalar into a scalar_tensor.
    Otherwise, after to_edge, the triu operation will be decomposed into several operations that include aten.ge.Scalar.
    """

    def __init__(self) -> None:
        super().__init__()

    def _replace_output(
        self, node: torch.fx.Node, output_node: torch.fx.Node, remap: Dict
    ):
        for user in node.users.copy():
            # remap
            user.replace_input_with(
                node,
                remap[output_node.args[0]],
            )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        decom_mappings = get_decompositions([torch.ops.aten.triu.default])

        for node in graph.nodes:
            if node.target == torch.ops.aten.triu.default:
                input_args = [node.args[0].meta["val"]]
                # The args[1], diagonal, is optional
                if len(node.args) > 1:
                    input_args.append(node.args[1])
                decomposed_module = make_fx(
                    node.target,
                    decomposition_table=decom_mappings,
                    tracing_mode="fake",
                )(*input_args)

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {}
                    remap["arg0_1"] = node.args[0]

                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                        predicate=lambda decomp_node: "arg1_1" not in decomp_node.name,
                        output_processor=self._replace_output,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
