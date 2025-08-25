# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.pass_base import ExportPass, PassResult


# TODO: Remove this workaround once HTP fixes the bug — LayerNorm without weights should be supported.
class InsertFrozenLayerNormWeight(ExportPass):
    """
    This pass injects a frozen weight parameter (filled with ones) into LayerNorm ops
    that were exported without weight (i.e., elementwise_affine=False), to satisfy
    backends that require the presence of a weight parameter.

    It operates at the ExportedProgram level, modifying both the FX graph and
    the graph_signature to include the new frozen parameter.

    Example transformation:

        Before:
            %out = aten.layer_norm(%x, normalized_shape=[128], weight=None, bias=None, eps=1e-5)

        After:
            %weight = get_attr("layer_norm_weight_0")
            %out = aten.layer_norm(%x, normalized_shape=[128], weight=%weight, bias=None, eps=1e-5)

    The injected weight is a frozen parameter with all values set to 1.0.
    """

    def __init__(self):
        super(InsertFrozenLayerNormWeight, self).__init__()
        self.layer_norm = torch.ops.aten.layer_norm.default

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        frozen_weight_idx = 0

        for node in graph.nodes:
            if node.op != "call_function" or node.target != self.layer_norm:
                continue

            # Detect LayerNorm ops missing the 'weight' argument
            if len(node.args) < 3:
                normalized_shape = node.args[1]

                # Create a frozen weight tensor filled with ones
                param_name = f"{self.layer_norm.__name__.split('.')[0]}_weight_{frozen_weight_idx}"
                frozen_weight = torch.ones(normalized_shape)
                graph_module.register_buffer(param_name, frozen_weight)
                with graph.inserting_before(node):
                    weight_node = graph.get_attr(param_name)
                node.args = (node.args[0], node.args[1], weight_node, *node.args[3:])

                frozen_weight_idx += 1
                modified = True

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, modified)
