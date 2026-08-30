# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FuseConsecutiveReshape(ExportPass):
    "For Demo Purpose Only, could possibly be done in other passes."

    def __init__(self):
        super().__init__()
        self.view = exir_ops.edge.aten.view_copy.default

    def _fuse(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target != self.view:
                continue
            src = node.args[0]
            while isinstance(src, torch.fx.Node) and src.target == self.view:
                src = src.args[0]
            if src is not node.args[0]:
                node.args = (src, *node.args[1:])

    def call(self, graph_module: torch.fx.GraphModule):
        self._fuse(graph_module)
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
