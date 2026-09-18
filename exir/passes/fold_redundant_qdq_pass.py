# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q
from torch.fx.passes.infra.pass_base import PassResult


class FoldRedundantDequantizeQuantizePass(ExportPass):
    """Fold adjacent ``dequantize -> quantize`` pairs using the shared qparam matcher.

    Decomposition can erase a quantized no-op, such as eval-mode dropout, leaving
    its surrounding dequantize and quantize nodes adjacent.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        dequant_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target in _DEQUANT_OPS
        ]

        num_nodes_before = len(graph_module.graph.nodes)
        eliminate_dq_q(graph_module, dequant_nodes)
        graph_module.graph.eliminate_dead_code()
        modified = len(graph_module.graph.nodes) != num_nodes_before
        if modified:
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
