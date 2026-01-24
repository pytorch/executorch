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
            if node.op == "call_function" and node.target in {
                torch.ops.aten.reciprocal.default,
            }:
                reciprocal_node = node
                reciprocal_node_input = node.args[0]

                # Create tensor of ones with same shape and dtype as input
                fake_val = reciprocal_node_input.meta["val"]
                ones_tensor = torch.ones(*fake_val.size(), dtype=fake_val.dtype)

                # Generate unique name and register buffer
                buffer_name = get_new_attr_name_with_prefix("_ones_constant_")(
                    graph_module
                )
                graph_module.register_buffer(buffer_name, ones_tensor)

                with graph_module.graph.inserting_after(reciprocal_node_input):
                    # Create get_attr node for the ones tensor
                    ones_node = graph.get_attr(buffer_name)
                    ones_node.meta = copy_meta(reciprocal_node.meta)

                    with graph_module.graph.inserting_after(ones_node):
                        # Create division node: ones / input
                        div_node = graph.call_function(
                            torch.ops.aten.div.Tensor,
                            (ones_node, reciprocal_node_input),
                        )
                    div_node.meta = copy_meta(reciprocal_node.meta)

                    # Replace all uses of reciprocal with division
                    for user in reciprocal_node.users.copy():
                        user.replace_input_with(reciprocal_node, div_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
