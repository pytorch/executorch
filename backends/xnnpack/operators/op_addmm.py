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

from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_FLAG_TRANSPOSE_WEIGHTS,
)


@register_node_visitor
class AddmmVisitor(NodeVisitor):
    target = "aten.addmm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        input_type_map = InputTypeToIndex(node_input=1, node_weight=2, node_bias=0)
        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, input_type_map=input_type_map
        )

        # bias
        bias_id = vals_to_ids[get_input_node(node, 0)]

        # input
        input_id = vals_to_ids[get_input_node(node, 1)]

        # filter
        filter_id = vals_to_ids[get_input_node(node, 2)]

        # output
        output_id = vals_to_ids[node]

        flag = XNN_FLAG_TRANSPOSE_WEIGHTS

        ser_node = XNode(
            xnode_union=XNNFullyConnected(
                input1_id=input_id,
                filter_id=filter_id,
                bias_id=bias_id,
                output_id=output_id,
                flags=flag,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
