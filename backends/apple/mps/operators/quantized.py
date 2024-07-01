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
    MPSGraph,
    MPSInt8PackedMM,
)

@register_node_visitor
class QuantizedLinearVisitor(NodeVisitor):
    target = ["aten._weight_int8pack_mm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_graph.mps_nodes.append(self.create_tertiary_node(node, mps_graph, MPSInt8PackedMM))
