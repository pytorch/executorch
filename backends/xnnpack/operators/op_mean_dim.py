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
    XNNGlobalAvgPooling2d,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class MeanDim(NodeVisitor):
    """
    XNNPACK only supports a special case of mean dim in which the operation can be written
    as Global Average Pooling. In order to be handled by xnnpack the input tensor must be 4d,
    the dimensions to reduce must be the two innermost (-1, -2) or (-2, -1). and the flag
    for keepdim must be set to True.
    """

    target = "aten.mean.dim"

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

        # mean dims
        mean_dims = cast(List[int], node.args[1])
        check_or_raise(
            mean_dims == [-1, -2] or mean_dims == [-2, -1],
            "XNNPACK only supports mean.dim across the innermost dimensions",
        )

        # keep dims
        check_or_raise(
            len(node.args) == 3 and bool(node.args[2]),
            "XNNPACK only supports mean.dim that keeps dims",
        )

        input_shape = get_tensor_value(xnn_graph.xvalues[input_id]).dims
        check_or_raise(
            len(input_shape) == 4, "Require input to mean.dim be 4 dimensional"
        )

        ser_node = XNode(
            xnode_union=XNNGlobalAvgPooling2d(
                input_id=input_id, output_id=output_id, flags=XNN_FLAG_KEEP_DIMS
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
