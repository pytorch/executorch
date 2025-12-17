# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.exir.pass_base import PassResult


class PropagateCustomMetaPass(XNNPACKPass):
    """
    Pass to propagate node.meta['custom'] from parent nodes to their q/dq child nodes.
    For all quantize/dequantize nodes in the graph, if the parent node has a
    node.meta['custom'] entry, this pass will copy that value to the q/dq node's meta.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        for node in graph.nodes:
            if not (is_quant(node) or is_dequant(node)):
                continue

            # Get the parent node (first input argument)
            if len(node.all_input_nodes) == 0:
                continue

            parent_node = node.args[0]
            if not isinstance(parent_node, torch.fx.Node):
                continue

            if "custom" in parent_node.meta:
                node.meta["custom"] = parent_node.meta["custom"]

        graph_module.recompile()

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
