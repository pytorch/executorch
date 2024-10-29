# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ConvertInterpolateWithUpsample2D(ExportPass):
    """
    Merge decomposed operators from interpolate back to one super node.
    TODO: Currently we only map to upsample2d version, should extend the
    capability by reverse engineering the decomposition process.
    """

    def __init__(self):
        super(ConvertInterpolateWithUpsample2D, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.functional.interpolate])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                input_node = src_partition.input_nodes[0]
                output_node = src_partition.output_nodes[0]
                with graph.inserting_after(input_node):
                    # TODO: robust way to get the configuration parameters and operator
                    # please check torch/_decomp/decomposition.py for details
                    if output_node.target.__name__ == "aten.index.Tensor":
                        # nearest_2d
                        # args: input, output_size, scales_h, scales_w
                        output_size = list(output_node.meta["val"].shape)
                        args = [input_node, output_size[-2:]]
                        upsample_op = exir_ops.edge.aten.upsample_nearest2d.default
                    else:
                        # upsample_2d
                        # args: input, output_size, aligned_corners, scales_h, scales_w
                        output_size = list(output_node.meta["val"].shape)
                        args = [input_node, output_size[-2:], False]
                        upsample_op = exir_ops.edge.aten.upsample_bilinear2d.default

                    upsample2d_node = graph.create_node(
                        "call_function", upsample_op, tuple(args)
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, upsample2d_node)
                    # copy metadata
                    upsample2d_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
