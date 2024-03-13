#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSArange,
    MPSGraph,
    MPSNode,
)
from executorch.backends.apple.mps.utils.mps_utils import edge_dtype_to_mps_dtype


@register_node_visitor
class ArangeVisitor(NodeVisitor):
    target = "aten.arange.start_step"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        step = 1.0
        if len(node.args) > 2 and node.args[2] is not None:
            step = float(node.args[2])

        start = float(node.args[0])
        end = float(node.args[1])

        dtype = edge_dtype_to_mps_dtype(node.meta["val"].dtype)
        if node.kwargs and "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
            dtype = edge_dtype_to_mps_dtype(node.kwargs["dtype"])

        output_id = self.define_tensor(node, mps_graph)

        mps_node = MPSNode(
            mpsnode_union=MPSArange(
                output_id=output_id,
                start=start,
                end=end,
                step=step,
                dtype=dtype,
            )
        )
        mps_graph.mps_nodes.append(mps_node)
