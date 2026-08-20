# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
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
            # Support if the rank of input tensor: {input_dims} is less than the rank of output tensor: {output_dims}.
            exir_ops.edge.aten.expand_copy.default,
        ]

    def traverse_broadcast_node(
        self, graph_module: torch.fx.GraphModule, reshape_cache
    ):
        for node in graph_module.graph.nodes:
            if node.target in self.broadcast_op_targets:
                for arg in node.args:
                    if not isinstance(arg, torch.fx.Node):
                        continue
                    input_rank = len(arg.meta["val"].shape)
                    output_rank = len(node.meta["val"].shape)
                    if input_rank != output_rank:
                        new_rank = [1] * (output_rank - input_rank) + list(
                            arg.meta["val"].shape
                        )
                        # Redirect ONLY the current broadcast node's input to the reshaped
                        # view, not every user of `arg`. Rewriting all users leaks the rank
                        # promotion into unrelated consumers (e.g. an in-place mutation of a
                        # rank-0 user input), producing a rank-mismatched USER_INPUT_MUTATION
                        # write-back that fails in to_executorch().
                        # Dedupe reshapes by (arg, new_rank) so that multiple broadcast ops
                        # sharing the same operand and target rank reuse a single view_copy
                        # instead of each creating their own.
                        cache_key = (arg, tuple(new_rank))
                        reshape_node = reshape_cache.get(cache_key)
                        if reshape_node is None:
                            with graph_module.graph.inserting_after(arg):
                                reshape_node = graph_module.graph.create_node(
                                    "call_function",
                                    exir_ops.edge.aten.view_copy.default,
                                    (arg, tuple(new_rank)),
                                )
                                # try skip dq_ops to get correct param node if applicable
                                arg_meta = (
                                    arg.args[0].meta
                                    if arg.target in dq_ops
                                    else arg.meta
                                )
                                # meta needs to be copied elementwisely for fake-tensor
                                # to be updated correctly and not affect meta of arg
                                for k, v in arg_meta.items():
                                    reshape_node.meta[k] = v
                                reshape_node.meta["val"] = reshape_node.meta[
                                    "val"
                                ].reshape(new_rank)
                            reshape_cache[cache_key] = reshape_node
                        node.replace_input_with(arg, reshape_node)

    def call(self, graph_module: torch.fx.GraphModule):
        self.traverse_broadcast_node(graph_module, {})
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
