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
    MPSAddmm,
    MPSGraph,
    MPSMatMul,
)


@register_node_visitor
class MatMulVisitor(NodeVisitor):
    target = ["aten.mm.default", "aten.bmm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_graph.mps_nodes.append(self.create_binary_node(node, mps_graph, MPSMatMul))


@register_node_visitor
class AddmmVisitor(NodeVisitor):
    target = "aten.addmm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_tertiary_node(node, mps_graph, MPSAddmm)

        if len(node.args) == 4:
            mps_node.mpsnode_union.beta = node.args[3]
        if len(node.args) == 5:
            mps_node.mpsnode_union.alpha = node.args[4]

        mps_graph.mps_nodes.append(mps_node)
