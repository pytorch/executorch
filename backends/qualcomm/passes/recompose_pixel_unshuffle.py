# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class RecomposePixelUnshuffle(ExportPass):
    """
    Merge decomposed operators from mathematically equivalent implementation
    back to one super node.
    """

    def __init__(self, quantization_capture=False):
        super(RecomposePixelUnshuffle, self).__init__()
        self.reshape_target = exir_ops.edge.aten.view_copy.default
        self.permute_target = exir_ops.edge.aten.permute_copy.default
        self.view_target = exir_ops.edge.aten.view_copy.default
        self.op = exir_ops.edge.aten.pixel_unshuffle.default

        self.quantization_capture = quantization_capture
        if quantization_capture:
            self.reshape_target = torch.ops.aten.reshape.default
            self.permute_target = torch.ops.aten.permute.default
            self.view_target = torch.ops.aten.view.default
            self.op = torch.ops.aten.pixel_unshuffle.default

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # math equivalent implementation
        for node in graph.nodes:
            if node.op == "call_function" and node.target == self.reshape_target:
                with graph.inserting_after(node):
                    premute_node = node.args[0]
                    if any(
                        [
                            len(node.args[1]) != 4,
                            premute_node.op != "call_function",
                            premute_node.target != self.permute_target,
                        ]
                    ):
                        continue

                    view_node = premute_node.args[0]
                    if any(
                        [
                            view_node.op != "call_function",
                            view_node.target != self.view_target,
                            len(view_node.args[1]) != 6,
                            len(premute_node.args[1]) != 6,
                        ]
                    ):
                        continue

                    b_in, d_nominal, h_in, s_h, w_in, s_w = view_node.args[1]
                    b_out, d_out, w_out, h_out = node.args[1]
                    if any(
                        [
                            b_out != b_in,
                            d_out != d_nominal * s_h * s_w,
                            w_in != w_out,
                            h_in != h_out,
                        ]
                    ):
                        continue

                    input_node = view_node.args[0]
                    args = (input_node, s_h)
                    pixel_unshuffle_node = graph.create_node(
                        "call_function", self.op, args
                    )
                    users = node.users.copy()
                    for user in users:
                        user.replace_input_with(node, pixel_unshuffle_node)
                    # copy metadata
                    pixel_unshuffle_node.meta = node.meta

        # decomposed core aten ops
        if not self.quantization_capture:
            partitions = get_source_partitions(graph, [torch.nn.PixelUnshuffle])
            for _, src_partitions in partitions.items():
                for src_partition in src_partitions:
                    input_node = src_partition.input_nodes[0]
                    output_node = src_partition.output_nodes[0]
                    with graph.inserting_after(input_node):
                        h_in_shape = input_node.meta["val"].shape[2]
                        h_out_shape = output_node.meta["val"].shape[2]
                        downscale_factor = h_in_shape / h_out_shape

                        op = self.op
                        pixel_unshuffle_node = graph.create_node(
                            "call_function", op, (input_node, int(downscale_factor))
                        )
                        users = output_node.users.copy()
                        for user in users:
                            user.replace_input_with(output_node, pixel_unshuffle_node)
                        # copy metadata
                        pixel_unshuffle_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
