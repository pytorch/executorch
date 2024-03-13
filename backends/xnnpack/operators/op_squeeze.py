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
    XNNStaticReshape,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class SqueezeVisitor(NodeVisitor):
    target = "aten.squeeze_copy.dim"

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
            cast(int, node.args[1]) == -1,
            "XNNPACK currently only supports squeezing in last dimension",
        )

        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)
        input_node = get_input_node(node, 0)

        # input
        input_id = vals_to_ids[input_node]

        # output
        output_id = vals_to_ids[node]

        check_or_raise(
            "val" in input_node.meta,
            "Missing val in tensor metadata for input when serializing XNNStaticReshape node",
        )
        new_shape = get_shape(input_node)[:-1]

        ser_node = XNode(
            xnode_union=XNNStaticReshape(
                num_dims=len(new_shape),
                new_shape=new_shape,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)


@register_node_visitor
class UnsqueezeVisitor(NodeVisitor):
    target = "aten.unsqueeze_copy.default"

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
            cast(int, node.args[1]) == -1,
            "XNNPACK currently only supports unsqueezing in last dimension",
        )

        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)
        input_node = get_input_node(node, 0)

        # input
        input_id = vals_to_ids[input_node]

        # output
        output_id = vals_to_ids[node]

        check_or_raise(
            "val" in input_node.meta,
            "Missing val in tensor metadata for input when serializing XNNStaticReshape node",
        )
        new_shape = get_shape(input_node) + [1]

        ser_node = XNode(
            xnode_union=XNNStaticReshape(
                num_dims=len(new_shape),
                new_shape=new_shape,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
