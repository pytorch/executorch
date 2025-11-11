# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNAvgPooling2d,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class AveragePooling2d(NodeVisitor):
    target = "aten.avg_pool2d.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, convert_to_nhwc=True
        )

        # input
        input_id = vals_to_ids[cast(torch.fx.Node, node.args[0])]

        # output
        output_id = vals_to_ids[node]

        # kernel_size
        pooling_height, pooling_width = cast(List, node.args[1])

        # stride
        stride_height, stride_width = cast(List, node.args[2])

        # padding
        padding_height, padding_width = 0, 0
        if node.args[3] is not None:
            padding_height, padding_width = cast(List[int], node.args[3])

        ser_node = XNode(
            xnode_union=XNNAvgPooling2d(
                padding_top=padding_height,
                padding_right=padding_width,
                padding_bottom=padding_height,
                padding_left=padding_width,
                pooling_height=pooling_height,
                pooling_width=pooling_width,
                stride_height=stride_height,
                stride_width=stride_width,
                dilation_height=0,  # Unused
                dilation_width=0,  # Unused
                input_id=input_id,
                output_id=output_id,
                flags=XNN_FLAG_KEEP_DIMS,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
