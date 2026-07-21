# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q
from torch.fx.passes.infra.pass_base import PassResult


class FoldRedundantDequantizeQuantizePass(NeutronEdgePass):
    """Fold redundant ``dequantize -> quantize`` pairs with identical qparams.

    A dequantize immediately followed by a quantize at identical qparams is the
    identity on the already-quantized value, so this pass reuses the shared
    ``eliminate_dq_q`` helper to rewire each such quantize's consumers to the
    dequantize's quantized input, removing the island and letting the neighboring
    clusters delegate as a single subgraph.
    """

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        dequant_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target in _DEQUANT_OPS
        ]

        num_nodes_before = len(graph_module.graph.nodes)
        eliminate_dq_q(graph_module, dequant_nodes)
        graph_module.graph.eliminate_dead_code()
        modified = len(graph_module.graph.nodes) != num_nodes_before

        return PassResult(graph_module, modified)
