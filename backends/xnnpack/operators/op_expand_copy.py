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
    XNNExpandDims,
    XNNGraph,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import get_input_node


def check_expand_copy_constraints(node: torch.fx.Node) -> bool:
    """
    Checks whether the given expand_copy node is delegatable to XNNPACK.
    XNNPACK only allows insertion of size-1 dimensions, not expanding existing
    dims.
    """
    in_shape = get_input_node(node, 0).meta["val"].shape
    new_shape = list(node.args[1])

    assert len(new_shape) >= len(
        in_shape
    ), "Expanded shape must have rank >= input rank."

    # Check new leading dims (if any). They must be of size 1.
    new_leading_dims_count = len(new_shape) - len(in_shape)
    for i in range(new_leading_dims_count):
        if new_shape[i] != 1:
            return False

    # Check existing dims. PyTorch expand semantics don't allow for dim insertion other
    # than at the front, so we just need to make sure none of the dims are expanded.
    for i in range(len(new_shape) - new_leading_dims_count):
        new_shape_at_dim = new_shape[new_leading_dims_count + i]
        # -1 means preserve dim.
        if new_shape_at_dim != -1 and new_shape_at_dim != in_shape[i]:
            return False

    return True


def get_inserted_dim_indices(
    node: torch.fx.Node,
) -> list[int]:
    """
    Returns the indices of the inserted dimensions in the expanded shape. Assumes that
    the node meets the conditions checked in check_expand_copy_constraints.
    """
    in_shape = get_input_node(node, 0).meta["val"].shape
    new_shape = list(node.args[1])
    new_dim_indices = []

    assert len(new_shape) >= len(
        in_shape
    ), "Expanded shape must have rank >= input rank."

    # PyTorch expand semantics enforce new dim insertion only at the front.
    new_leading_dims_count = len(new_shape) - len(in_shape)
    for i in range(new_leading_dims_count):
        if new_shape[i] != 1:
            return False
        else:
            new_dim_indices.append(i)

    return new_dim_indices


@register_node_visitor
class ExpandCopyVisitor(NodeVisitor):
    target = "aten.expand_copy.default"

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

        # output
        output_id = vals_to_ids[node]

        new_dim_indices = get_inserted_dim_indices(node)

        ser_node = XNode(
            xnode_union=XNNExpandDims(
                num_new_dims=len(new_dim_indices),
                new_dim_indices=new_dim_indices,
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
