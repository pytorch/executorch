# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    XNNGraph,
    XNNStaticReshape,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_input_node,
    PERM_NCHW_TO_NHWC,
)


@register_node_visitor
class ViewCopyVisitor(NodeVisitor):
    target = "aten.view_copy.default"

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
            "Missing val in tensor metadata for input when serializing XNNStaticReshape",
        )

        # output shape
        check_or_raise(
            "val" in node.meta,
            "Missing val in tensor metadata for input when serializing XNNStaticReshape",
        )

        new_shape = node.args[1]
        check_or_raise(
            all(isinstance(d, int) for d in new_shape),
            "Symbolic reshape parameter is not supported in XNNStaticReshape",
        )

        # PyTorch uses -1 for inferred dims, whereas XNNPACK expects 0.
        new_shape = tuple(d if d != -1 else 0 for d in new_shape)

        # Handle NCHW dim order - if this op is in NCHW order, we need to permute the
        # view shape correspondingly.
        if "XNN_NHWC_NODE" in node.meta:
            check_or_raise(len(new_shape) == 4, "Invalid NCHW shape")
            new_shape = [new_shape[PERM_NCHW_TO_NHWC[n]] for n in range(4)]

        num_dynamic_dims = sum(1 for d in new_shape if d == 0)

        check_or_raise(
            num_dynamic_dims <= 1,
            "XNNPACK reshape only supports 1 dynamic dimension.",
        )

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
