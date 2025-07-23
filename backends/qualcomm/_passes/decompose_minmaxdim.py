# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
from collections import Counter

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class DecomposeMinMaxDim(ExportPass):
    """
    Since QNN does not support multi-output ops, this pass decomposes
        `torch.min(dim=...)` and `torch.max(dim=...)` into two separate operations:
        - `aten.min.dim` / `aten.max.dim` for the value
        - `aten.argmin` / `aten.argmax` for the index

        Example transformation in the exported FX graph:

            Python source:
                val, idx = torch.min(x, dim=1)

            Before:
                %min = aten.min(%x, dim=1)
                %val = getitem(%min, 0)
                %idx = getitem(%min, 1)

            After:
                %min = aten.min(%x, dim=1)
                %val = getitem(%min, 0)
                %idx = aten.argmin(%x, dim=1)

    This pass preserves the value output if used, and transforms only the index path.
    """

    def __init__(self):
        super().__init__()
        self.min_dim = exir_ops.edge.aten.min.dim
        self.max_dim = exir_ops.edge.aten.max.dim
        self.argmin = exir_ops.edge.aten.argmin.default
        self.argmax = exir_ops.edge.aten.argmax.default
        self.getitem = operator.getitem

        # index-only op
        self.replace_table = {
            self.min_dim: self.argmin,
            self.max_dim: self.argmax,
        }

        self.patterns = [
            # Only index is used (e.g., _, idx = torch.min(x, dim=1))
            {self.min_dim: 1, self.getitem: 1},
            {self.max_dim: 1, self.getitem: 1},
            # Both value and index are used (e.g., val, idx = torch.max(x, dim=1))
            {self.min_dim: 1, self.getitem: 2},
            {self.max_dim: 1, self.getitem: 2},
        ]

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.min, torch.max])
        for _, src_partitions in partitions.items():
            for partition in src_partitions:
                if Counter([n.target for n in partition.nodes]) not in self.patterns:
                    continue
                binary_output_node = partition.nodes[0]

                # Ensure the binary-output node has exactly 2 outputs
                if len(binary_output_node.meta["val"]) != 2:
                    continue

                input_tensor = binary_output_node.args[0]
                dim = binary_output_node.args[1]
                keepdim = (
                    binary_output_node.args[2]
                    if len(binary_output_node.args) > 2
                    else False
                )

                idx_node = next(
                    (
                        output_node
                        for output_node in partition.output_nodes
                        if output_node.meta["val"].dtype == torch.int64
                    ),
                    None,
                )

                if idx_node:
                    with graph.inserting_before(idx_node):
                        argmin_node = graph.create_node(
                            "call_function",
                            self.replace_table[binary_output_node.target],
                            (input_tensor, dim, keepdim),
                        )
                        argmin_node.meta = idx_node.meta

                    for user in list(idx_node.users):
                        user.replace_input_with(idx_node, argmin_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
