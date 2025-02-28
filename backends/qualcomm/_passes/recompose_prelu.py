# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class RecomposePReLU(ExportPass):
    """
    Merge decomposed operators from prelu back to one super node.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(RecomposePReLU, self).__init__()
        self.edge_program = edge_program

    def _get_coeff_node(self, nodes: List[torch.fx.Node]):
        for node in nodes:
            if node.target == exir_ops.edge.aten.view_copy.default:
                return node.args[0]

    def _get_input_node(self, nodes: List[torch.fx.Node], coeff_node):
        return [n for n in nodes if n != coeff_node][0]

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.PReLU, torch.nn.LeakyReLU])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                # somehow op might not be decomposed, skip it
                if len(src_partition.nodes) == 1:
                    continue

                coeff_node = self._get_coeff_node(src_partition.nodes)
                input_node = self._get_input_node(src_partition.input_nodes, coeff_node)
                output_node = src_partition.output_nodes[0]

                with graph.inserting_before(output_node):
                    prelu_op = exir_ops.edge.aten.prelu.default
                    prelu_node = graph.create_node(
                        "call_function", prelu_op, (input_node, coeff_node)
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, prelu_node)
                    # copy metadata
                    prelu_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
