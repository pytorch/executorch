# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    check_or_raise,
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNMaxPooling2d,
    XNode,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class MaxPooling2d(NodeVisitor):
    target = "aten.max_pool2d.default"

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
        kwargs = {}

        kwargs["input_id"] = vals_to_ids[node.all_input_nodes[0]]

        input_shape = get_tensor_value(xnn_graph.xvalues[kwargs["input_id"]]).dims
        check_or_raise(len(input_shape) == 4, "Require input to be 4 dimensional")

        # output
        kwargs["input_id"] = vals_to_ids[node.all_input_nodes[0]]
        kwargs["output_id"] = vals_to_ids[node]

        # kernel info
        kernal_shape = cast(List[int], node.args[1])
        kwargs["pooling_height"] = kernal_shape[0]
        kwargs["pooling_width"] = kernal_shape[1]

        # stride info
        stride = cast(List[int], node.args[2])
        kwargs["stride_height"] = stride[0]
        kwargs["stride_width"] = stride[1]

        # padding info
        kwargs["padding_top"] = 0
        kwargs["padding_right"] = 0
        kwargs["padding_bottom"] = 0
        kwargs["padding_left"] = 0

        if len(node.args) > 3:
            padding_shape = cast(List[int], node.args[3])
            kwargs["padding_top"] = padding_shape[0]
            kwargs["padding_right"] = padding_shape[1]
            kwargs["padding_bottom"] = padding_shape[0]
            kwargs["padding_left"] = padding_shape[1]

        # dilation info
        kwargs["dilation_height"] = 1
        kwargs["dilation_width"] = 1
        if len(node._args) > 4:
            dilation = cast(List[int], node.args[4])
            kwargs["dilation_height"] = dilation[0]
            kwargs["dilation_width"] = dilation[1]

        kwargs["flags"] = XNN_FLAG_KEEP_DIMS

        ser_node = XNode(
            xnode_union=XNNMaxPooling2d(
                **kwargs,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
