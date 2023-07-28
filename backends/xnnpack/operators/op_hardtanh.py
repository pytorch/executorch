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
from executorch.backends.xnnpack.utils.utils import get_input_node


@register_node_visitor
class HardTanhVisitor(NodeVisitor):
    target = "aten.hardtanh.default"

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

        # default value of output_min and output_max
        output_min = -1
        output_max = 1

        if len(node.args) > 1:
            # update output_min
            output_min = cast(float, node.args[1])

            # update output_max
            output_max = cast(float, node.args[2])

        # input_id
        input_id = vals_to_ids[get_input_node(node, 0)]

        # output
        output_id = vals_to_ids[node]

        output_min_max = OutputMinMax(output_min=output_min, output_max=output_max)

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
