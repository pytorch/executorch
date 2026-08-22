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
    XNNGraph,
    XNNSoftmax,
    XNode,
)
from executorch.backends.xnnpack.utils.utils import check_or_raise, get_input_node


@register_node_visitor
class SoftmaxVisitor(NodeVisitor):
    target = "aten._softmax.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        xnn_graph: XNNGraph,
        vals_to_ids: Dict[torch.fx.Node, int],
        debug_handle: int,
    ) -> None:
        # XNNPACK does not support softmax_dim != -1, atleast from the graph level APIs.
        # XNNPACK partitioner should not let this pass, let's just make sure.
        softmax_dim = node.args[1]
        input_dim = get_input_node(node, 0).meta["val"].dim()
        check_or_raise(
            bool(softmax_dim == -1) or bool(softmax_dim == input_dim - 1),
            f"XNNPACK does not support softmax_dim != -1, but got {softmax_dim} for tensor with dim() = {input_dim}",
        )

        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)

        # input
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        ser_node = XNode(
            xnode_union=XNNSoftmax(input_id=input_id, output_id=output_id, flags=0),
            debug_handle=debug_handle,
        )
        xnn_graph.xnodes.append(ser_node)
