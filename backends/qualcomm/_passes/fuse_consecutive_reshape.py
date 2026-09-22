# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.qualcomm.utils.constants import QCOM_REQUANTIZE
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FuseConsecutiveReshape(ExportPass):
    """
    Collapse chains of consecutive view_copy into a single view_copy.
    This condition is not added in remove_redundancy.py since
    quantized(QDQ) models are hard to handle, and remove_redundancy runs before FoldQDQ.
    Some reshape shows up at edge dialect, so annotation phase of remove_redundancy
    also won't work.
    This pass is designed to be conservative and will not fold if requantize meta is
    found during traversal. The only exception is when the requantize meta is on the
    last view_copy of the chain, since that node is the one being rewritten rather
    than a node being folded away.
    """

    def __init__(self):
        super().__init__()
        self.view = exir_ops.edge.aten.view_copy.default

    def _fuse(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target != self.view:
                continue
            # Walk back to the first non-view source node; every intermediate view
            # is redundant because this node restates the full target shape.
            # Need to ensure this optimization doesn't break requantize logic.
            src = original_src = node.args[0]
            while (
                isinstance(src, torch.fx.Node)
                and src.target == self.view
                and QCOM_REQUANTIZE not in src.meta
            ):
                src = src.args[0]

            if src is not original_src and QCOM_REQUANTIZE not in src.meta:
                node.args = (src, *node.args[1:])

    def call(self, graph_module: torch.fx.GraphModule):
        self._fuse(graph_module)
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
