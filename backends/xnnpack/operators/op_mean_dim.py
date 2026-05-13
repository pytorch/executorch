# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

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
from executorch.backends.xnnpack.utils.utils import normalize_mean_dims
from executorch.backends.xnnpack.utils.xnnpack_constants import XNN_FLAG_KEEP_DIMS


@register_node_visitor
class MeanDim(NodeVisitor):
    """
    XNNPACK only supports the special case of mean.dim that can be lowered
    to Global Average Pooling. The input tensor must be 4D, keepdim must be
    True, and the reduced dimensions must normalize to [2, 3] (for example
    [2, 3], [3, 2], [-1, -2], or [-2, -1]).
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

        input_shape = get_tensor_value(xnn_graph.xvalues[input_id]).dims
        check_or_raise(
            len(input_shape) == 4, "Require input to mean.dim be 4 dimensional"
        )

        # This visitor serializes mean.dim as Global Average Pooling, which has
        # no field for an explicit dtype override.
        check_or_raise(
            node.kwargs.get("dtype") is None,
            "XNNPACK does not support mean.dim with dtype",
        )

        # mean dims
        mean_dims = normalize_mean_dims(node.args[1], len(input_shape))
        check_or_raise(
            sorted(mean_dims) == [2, 3],
            "XNNPACK only supports mean.dim across the innermost dimensions",
        )

        # keep dims
        check_or_raise(
            len(node.args) == 3 and bool(node.args[2]),
            "XNNPACK only supports mean.dim that keeps dims",
        )

        ser_node = XNode(
            xnode_union=XNNGlobalAvgPooling2d(
                input_id=input_id, output_id=output_id, flags=XNN_FLAG_KEEP_DIMS
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
