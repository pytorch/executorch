# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ConvertPReLU(ExportPass):
    """
    Merge decomposed operators from prelu back to one super node.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(ConvertPReLU, self).__init__()
        self.edge_program = edge_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.PReLU])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                input_node = src_partition.input_nodes[0]
                output_node = src_partition.output_nodes[0]
                placeholders = [n for n in src_partition.nodes if n.op == "placeholder"]
                assert len(placeholders) == 1

                with graph.inserting_after(input_node):
                    prelu_op = exir_ops.edge.aten.prelu.default
                    prelu_node = graph.create_node(
                        "call_function", prelu_op, (input_node, placeholders[0])
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, prelu_node)
                    # copy metadata
                    prelu_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
