# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix

from .utils import copy_meta


class DecomposeReciprocal(ExportPass):
    def __init__(self):
        super(DecomposeReciprocal, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target in {
                torch.ops.aten.reciprocal.default,
            }:
                reciprocal_node = node
                reciprocal_node_input = node.args[0]
                with graph_module.graph.inserting_after(reciprocal_node_input):
                    # Create division node
                    div_node = graph.call_function(
                        torch.ops.aten.div.Tensor,
                        (1, reciprocal_node_input),
                    )
                div_node.meta = copy_meta(reciprocal_node.meta)

                # Replace all uses of reciprocal with division
                for user in reciprocal_node.users.copy():
                    user.replace_input_with(reciprocal_node, div_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
