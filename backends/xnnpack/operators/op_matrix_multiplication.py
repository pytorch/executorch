# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNFullyConnected,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node

from executorch.backends.xnnpack.utils.xnnpack_constants import (
    XNN_FLAG_TRANSPOSE_WEIGHTS,
    XNN_INVALID_VALUE_ID,
)


@register_node_visitor
class MatrixMultiplyVisitor(NodeVisitor):
    target = "aten.mm.default"

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

        # filter
        filter_id = vals_to_ids[get_input_node(node, 1)]

        # output
        output = vals_to_ids[node]

        # Matrix Multiply is handled by using linear with bias = 0. XNNPACK performs
        # this by giving a dummy id as the bias in the fully-connected node.
        ser_node = XNode(
            xnode_union=XNNFullyConnected(
                input1_id=input_id,
                filter_id=filter_id,
                bias_id=XNN_INVALID_VALUE_ID,  # Dummy Bias id for bias = 0
                output_id=output,
                # We are taking from Aten::mm which holds weights as (in, out)
                # instead of (out, in) which is what torch.nn.linear uses
                flags=XNN_FLAG_TRANSPOSE_WEIGHTS,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
