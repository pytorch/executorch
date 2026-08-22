#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

from typing import cast

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSClamp,
    MPSGraph,
    MPSMinMax,
    MPSWhere,
)


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target = "aten.clamp.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSClamp)

        min_value = "-inf"
        max_value = "inf"

        if len(node.args) >= 2 and node.args[1] is not None:
            min_value = cast(float, node.args[1])

        if len(node.args) >= 3 and node.args[2] is not None:
            max_value = cast(float, node.args[2])

        mps_node.min_max = MPSMinMax(min_value=min_value, max_value=max_value)
        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class WhereVisitor(NodeVisitor):
    target = "aten.where.self"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_graph.mps_nodes.append(self.create_tertiary_node(node, mps_graph, MPSWhere))
