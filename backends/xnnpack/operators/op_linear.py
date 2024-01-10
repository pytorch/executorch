# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    get_input_node,
    InputTypeToIndex,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNFullyConnected,
    XNNGraph,
    XNode,
)

from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_INVALID_VALUE_ID


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        input_type_map = (
            InputTypeToIndex(node_input=0, node_weight=1, node_bias=2)
            if len(node.args) == 3
            else InputTypeToIndex(node_input=0, node_weight=1)
        )
        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, input_type_map=input_type_map
        )

        # bias
        bias_id = (
            XNN_INVALID_VALUE_ID
            if len(node.args) == 2
            else vals_to_ids[get_input_node(node, input_type_map.node_bias)]
        )

        # input
        input_id = vals_to_ids[get_input_node(node, input_type_map.node_input)]

        # filter
        filter_id = vals_to_ids[get_input_node(node, input_type_map.node_weight)]

        # output
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNFullyConnected(
                input1_id=input_id,
                filter_id=filter_id,
                bias_id=bias_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
