# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    get_tensor_value,
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNArgMaxPooling2d,
    XNNGraph,
    XNNMaxPooling2d,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class MaxDim(NodeVisitor):
    target = "aten.amax.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:

        check_or_raise(
            len(node.args) == 3,
            "amax.default only supports keep_dim == True",
        )

        dim_val = cast(int, node.args[1])
        check_or_raise(
            dim_val == 2 or dim_val == 3,
            "amax.default only supports dim == 2 or dim == 3",
        )

        input_id = vals_to_ids[get_input_node(node, 0)]

        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, convert_to_nhwc=True
        )

        output_id = vals_to_ids[node]

        input_shape = get_tensor_value(xnn_graph.xvalues[input_id]).dims
        check_or_raise(
            len(input_shape) == 4, "Require input to max.dim be 4 dimensional"
        )

        # This is in NHWC
        pooling_height = 1
        pooling_width = 1
        stride_height = 1
        stride_width = 1
        if dim_val == 2:
            pooling_height = input_shape[1]
            pooling_width = 1
            stride_height = input_shape[1]
        elif dim_val == 3:
            pooling_height = 1
            pooling_width = input_shape[2]
            stride_width = input_shape[2]

        ser_node = XNode(
            xnode_union=XNNMaxPooling2d(
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
                padding_left=0,
                pooling_height=pooling_height,
                pooling_width=pooling_width,
                stride_height=stride_height,
                stride_width=stride_width,
                dilation_height=1,
                dilation_width=1,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )

        xnn_graph.xnodes.append(ser_node)


@register_node_visitor
class ArgMaxDim(NodeVisitor):
    target = "aten.max.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:

        check_or_raise(
            len(node.args) == 3,
            "max.dim only supports keep_dim == True",
        )

        dim_val = cast(int, node.args[1])
        check_or_raise(
            dim_val == 2 or dim_val == 3,
            "max.dim only supports dim == 2 or dim == 3",
        )

        # node.meta["val"] is a tuple (values_tensor, indices_tensor)
        # We don't care about how it is defined, so we can adjust val to be a
        # single tensor rather than a tuple arbitrarily just to make
        # define_nodes_tensor_inputs_outputs work
        original_val = node.meta["val"]
        node.meta["val"] = original_val[0]

        self.define_nodes_tensor_inputs_outputs(
            node, xnn_graph, vals_to_ids, convert_to_nhwc=True
        )
        for user in node.users:
            self.define_nodes_tensor_inputs_outputs(
                user, xnn_graph, vals_to_ids, convert_to_nhwc=True
            )

        # Restore node.meta["val"]
        node.meta["val"] = original_val

        input_id = vals_to_ids[get_input_node(node, 0)]

        input_shape = get_tensor_value(xnn_graph.xvalues[input_id]).dims
        check_or_raise(
            len(input_shape) == 4, "Require input to max.dim be 4 dimensional"
        )

        users = list(node.users.keys())

        if len(users) != 2:
            raise AssertionError(
                f"Invalid number of users for max.dim (Expected 2, Got: {len(users)})"
            )

        values_node = None
        indices_node = None

        for getitem_node in users:
            taget_name = cast(torch._ops.OpOverload, getitem_node.target).__name__
            if taget_name != "getitem":
                raise AssertionError(
                    f"Expected max node's user to be getitem, got: {taget_name}"
                )

            if getitem_node.args[1] == 0:
                values_node = getitem_node
            elif getitem_node.args[1] == 1:
                indices_node = getitem_node

        if values_node is None or indices_node is None:
            raise AssertionError(
                f"Expected max node's getitem args to be 1 and 2, got: {[user.args[1] for user in users]}"
            )

        output_index_id = vals_to_ids[indices_node]
        output_value_id = vals_to_ids[values_node]

        # This is in NHWC
        pooling_height = 1
        pooling_width = 1
        if dim_val == 2:
            pooling_height = input_shape[1]
            pooling_width = 1
        elif dim_val == 3:
            pooling_height = 1
            pooling_width = input_shape[2]

        ser_node = XNode(
            xnode_union=XNNArgMaxPooling2d(
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
                padding_left=0,
                pooling_height=pooling_height,
                pooling_width=pooling_width,
                input_id=input_id,
                output_value_id=output_value_id,
                output_index_id=output_index_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )

        xnn_graph.xnodes.append(ser_node)
