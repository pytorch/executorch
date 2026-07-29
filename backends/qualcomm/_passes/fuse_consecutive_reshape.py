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
    """
    Collapse chains of consecutive view_copy into a single view_copy. A
    view_copy is order-preserving, so ``view(view(x, A), B) == view(x, B)`` for
    any valid shapes; only the final shape matters.

    This matters for the ConvInplaceLinear -> ConvertMhaToSha interaction: the
    ConvertLinearToConv2d restore emits a rank-3 view_copy that sits between the
    conv and the head-making rank-4 view_copy. That intermediate view stops
    ConvertMhaToSha's _is_making_mha matcher from reaching the rank-4 reshape, so
    the Q/K/V projection weight is never split per head. Fusing the views away
    restores the conv -> permute -> view(4D) shape the matcher expects.
    """

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
