# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class ExpandBroadcastTensorShape(ExportPass):
    """
    Make tensors have same rank for layout-transform to work properly.
    """

    def __init__(self):
        super(ExpandBroadcastTensorShape, self).__init__()
        self.broadcast_op_targets = [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.div.Tensor,
        ]

    def traverse_broadcast_node(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target in self.broadcast_op_targets:
                for arg in node.args:
                    input_rank = len(arg.meta["val"].shape)
                    output_rank = len(node.meta["val"].shape)
                    if input_rank != output_rank:
                        with graph_module.graph.inserting_after(arg):
                            new_rank = [1] * (output_rank - input_rank) + list(
                                arg.meta["val"].shape
                            )
                            users = list(arg.users.keys())
                            reshape_node = graph_module.graph.create_node(
                                "call_function",
                                exir_ops.edge.aten.view_copy.default,
                                (arg, tuple(new_rank)),
                            )
                            # meta needs to be copied elementwisely for fake-tensor
                            # to be updated correctly and not affect meta of arg
                            for k, v in arg.meta.items():
                                reshape_node.meta[k] = v
                            reshape_node.meta["val"] = reshape_node.meta["val"].reshape(
                                new_rank
                            )
                            for user in users:
                                user.replace_input_with(arg, reshape_node)

    def call(self, graph_module: torch.fx.GraphModule):
        self.traverse_broadcast_node(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
