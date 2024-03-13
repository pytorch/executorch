# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNStaticSlice,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_input_node,
    PERM_NCHW_TO_NHWC,
    PERM_NHWC_TO_NCHW,
)


@register_node_visitor
class SliceCopyVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

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

        input_node = get_input_node(node, 0)

        # input
        input_id = vals_to_ids[input_node]

        # output
        output_id = vals_to_ids[node]

        # input shape
        check_or_raise(
            "val" in input_node.meta,
            "Missing val in tensor metadata for input when serializing XNNStaticSlice",
        )
        input_shape = get_shape(input_node)

        # output shape
        check_or_raise(
            "val" in node.meta,
            "Missing val in tensor metadata for input when serializing XNNStaticSlice",
        )
        output_shape = get_shape(node)
        dim_of_slice = cast(int, node.args[1])

        if "XNN_NHWC_NODE" in node.meta:
            input_shape = [input_shape[i] for i in PERM_NCHW_TO_NHWC]
            output_shape = [output_shape[i] for i in PERM_NCHW_TO_NHWC]
            dim_of_slice = PERM_NHWC_TO_NCHW[dim_of_slice]

        slice_begin_index = cast(int, node.args[2])
        if slice_begin_index < 0:
            slice_begin_index = input_shape[dim_of_slice] + slice_begin_index

        if len(node.args) > 4:
            stride = cast(int, node.args[4])
            check_or_raise(
                stride == 1, "XNNPACK Static Slice only supports slices with stride 1"
            )

        num_dims = len(input_shape)
        offsets = [0 for i in range(num_dims)]
        offsets[dim_of_slice] = slice_begin_index
        sizes = list(output_shape)

        ser_node = XNode(
            xnode_union=XNNStaticSlice(
                num_dims=num_dims,
                offsets=offsets,
                sizes=sizes,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
