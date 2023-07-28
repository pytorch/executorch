# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNLeakyReLU,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class LeakyReluVisitor(NodeVisitor):
    target = "aten.leaky_relu.default"

    # LeakyReLU nodes which use the default value for negative_slope don't have the
    # negative_slope value included in their args, so we need to hardcode it.
    # From https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE = 0.01

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)

        # input
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        # No negative_slope in args, meaning the default negative_slope is used
        negative_slope = (
            cast(float, node.args[1])
            if len(node.args) > 1
            else self.DEFAULT_LEAKY_RELU_NEGATIVE_SLOPE
        )

        ser_node = XNode(
            xnode_union=XNNLeakyReLU(
                negative_slope=negative_slope,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
