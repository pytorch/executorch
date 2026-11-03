# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


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

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # Match on the op rather than on source partitions: those need the
        # `torch.min` / `torch.max` source_fn, which `Tensor.max` and graphs
        # re-traced from a non-strict export do not carry.
        for node in list(graph.nodes):
            if node.target not in self.replace_table:
                continue
            idx_nodes = [
                user
                for user in node.users
                if user.target == self.getitem and user.args[1] == 1
            ]
            if not idx_nodes:
                continue
            input_tensor, dim = node.args[0], node.args[1]
            keepdim = node.args[2] if len(node.args) > 2 else False
            for idx_node in idx_nodes:
                with graph.inserting_before(idx_node):
                    arg_node = graph.create_node(
                        "call_function",
                        self.replace_table[node.target],
                        (input_tensor, dim, keepdim),
                    )
                    arg_node.meta = idx_node.meta
                idx_node.replace_all_uses_with(arg_node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
