# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class RemoveRedundancy(ExportPass):
    """
    Trim the 'identity' operators to reduce the unnecessary copy overhead.
    """

    redundant_ops = {
        torch.clone,
        torch.ops.aten.clone.default,
        exir_ops.edge.aten.clone.default,
        torch.ops.aten.alias.default,
        exir_ops.edge.aten.alias.default,
    }

    def __init__(self):
        super(RemoveRedundancy, self).__init__()

    def _remove(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if n.target not in self.redundant_ops:
                continue

            to_be_remove = n
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            graph_module.graph.erase_node(to_be_remove)

    def call(self, graph_module: torch.fx.GraphModule):
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
