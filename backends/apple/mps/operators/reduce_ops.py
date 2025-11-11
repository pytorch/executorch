#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import cast, List

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSGraph,
    MPSMean,
)


@register_node_visitor
class MeanVisitor(NodeVisitor):
    target = "aten.mean.dim"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSMean)

        dims = cast(List[int], node.args[1])
        mps_node.mpsnode_union.num_dims = len(dims)
        mps_node.mpsnode_union.dims = dims
        if len(node.args) == 3:
            mps_node.mpsnode_union.keep_dims = node.args[2]

        mps_graph.mps_nodes.append(mps_node)
