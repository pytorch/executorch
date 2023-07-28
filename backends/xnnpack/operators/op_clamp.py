# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.backends.xnnpack.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (
    OutputMinMax,
    XNNClamp,
    XNNGraph,
    XNode,
)


@register_node_visitor
class ReluVisitor(NodeVisitor):
    target = "aten.clamp.default"

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

        min_val = "-inf"
        max_val = "inf"

        if len(node.args) >= 2 and node.args[1] is not None:
            min_val = cast(float, node.args[1])

        if len(node.args) >= 3 and node.args[2] is not None:
            max_val = cast(float, node.args[2])

        # input_id
        input_id = vals_to_ids[node.all_input_nodes[0]]

        # output
        output_id = vals_to_ids[node]

        output_min_max = OutputMinMax(output_min=min_val, output_max=max_val)

        ser_node = XNode(
            xnode_union=XNNClamp(
                input_id=input_id,
                output_id=output_id,
                flags=0,
            ),
            debug_handle=debug_handle,
            output_min_max=output_min_max,
        )
        xnn_graph.xnodes.append(ser_node)
