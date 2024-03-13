# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class RecomposePixelShuffle(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self):
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # decomposed core aten ops
        partitions = get_source_partitions(graph, [torch.nn.PixelShuffle])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                input_node = src_partition.input_nodes[0]
                output_node = src_partition.output_nodes[0]
                with graph.inserting_after(input_node):
                    h_in_shape = input_node.meta["val"].shape[2]
                    h_out_shape = output_node.meta["val"].shape[2]
                    upscale_factor = h_out_shape / h_in_shape

                    pixel_shuffle_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.pixel_shuffle.default,
                        (input_node, int(upscale_factor)),
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, pixel_shuffle_node)
                    # copy metadata
                    pixel_shuffle_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
