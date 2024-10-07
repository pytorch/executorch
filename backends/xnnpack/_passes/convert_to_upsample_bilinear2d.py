# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.partition.graphs import bilinear_2d
from executorch.backends.xnnpack.utils.utils import check_or_raise
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher


class ConvertToUpsampleBilinear2d(XNNPACKPass):
    output_nodes_to_bilinear_node = {}

    def create_upsample_bilinear_2d(
        self,
        graph_module: torch.fx.GraphModule,
        internal_match: InternalMatch,
        align_corners: bool,
    ):
        output = internal_match.returning_nodes[0]
        output_shape = output.meta["val"].shape
        output_h = output_shape[-2]
        output_w = output_shape[-1]
        check_or_raise(
            isinstance(output_h, int) and isinstance(output_w, int),
            "XNNPACK Upsample Bilinear2d does not support dynamic shape",
        )

        input_node = internal_match.placeholder_nodes[-1]
        input_node = self.output_nodes_to_bilinear_node.get(input_node, input_node)
        with graph_module.graph.inserting_before(output):
            upsample_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.upsample_bilinear2d.vec,
                # TODO(T166527012): Using output_h and output_w here only works with static shapes
                args=(input_node, [output_h, output_w], align_corners, None),
            )
        output.replace_all_uses_with(upsample_node)
        self.output_nodes_to_bilinear_node[output] = upsample_node
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

    def call(self, graph_module: torch.fx.GraphModule):
        for pattern, align_corners in bilinear_2d.get_graphs_dict().items():
            sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
            matches = list(sm.match(graph_module.graph))
            for partition_to_replace in matches:
                self.create_upsample_bilinear_2d(
                    graph_module, partition_to_replace, align_corners
                )

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
