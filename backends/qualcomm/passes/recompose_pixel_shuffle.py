# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult


class RecomposePixelShuffle(ExportPass):
    """
    Merge decomposed operators from mathematically equivalent implementation
    back to one super node.
    """

    def __init__(self):
        super(RecomposePixelShuffle, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op == "call_function" and (
                node.target == torch.ops.aten.reshape.default
            ):
                with graph.inserting_after(node):
                    premute_node = node.args[0]
                    if any(
                        [
                            len(node.args[1]) != 4,
                            premute_node.op != "call_function",
                            premute_node.target != torch.ops.aten.permute.default,
                        ]
                    ):
                        continue

                    reshape_node = premute_node.args[0]
                    if any(
                        [
                            reshape_node.op != "call_function",
                            reshape_node.target != torch.ops.aten.reshape.default,
                            len(reshape_node.args[1]) != 6,
                        ]
                    ):
                        continue

                    b_in, d_nominal, blk, blk, w, h = reshape_node.args[1]
                    b_out, d_out, w_out, h_out = node.args[1]
                    if any(
                        [
                            b_out != b_in,
                            d_out != d_nominal,
                            w_out != w * blk,
                            h_out != h * blk,
                        ]
                    ):
                        continue

                    input_node = reshape_node.args[0]
                    args = (input_node, blk)
                    pixel_shuffle_node = graph.create_node(
                        "call_function", torch.ops.aten.pixel_shuffle.default, args
                    )
                    users = node.users.copy()
                    for user in users:
                        user.replace_input_with(node, pixel_shuffle_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
