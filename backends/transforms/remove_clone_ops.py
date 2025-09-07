# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q


class RemoveCloneOpsTransform(ExportPass):
    """
    Trim the 'identity' operators to reduce the unnecessary copy overhead.
    """

    clone_ops: Set[torch._ops.OpOverload] = {
        exir_ops.edge.aten.clone.default,
    }

    def __init__(self) -> None:
        super().__init__()

    def _remove(self, graph_module: torch.fx.GraphModule) -> None:
        dequant_nodes = []

        for n in graph_module.graph.nodes:
            if n.target not in self.clone_ops:
                continue

            to_be_remove = n
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            if n.args[0].target in _DEQUANT_OPS:
                dequant_nodes += [n.args[0]]
            graph_module.graph.erase_node(to_be_remove)

        eliminate_dq_q(graph_module, dequant_nodes)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
