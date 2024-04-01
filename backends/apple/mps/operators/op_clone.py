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
    MPSView,
)
from executorch.backends.transforms import get_shape
from executorch.exir.dialects._ops import ops as exir_ops


@register_node_visitor
class CloneVisitor(NodeVisitor):
    target = ["aten.clone.default", "aten._to_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        if node.target == exir_ops.edge.aten._to_copy.default:
            # TODO
            if len(node.args) > 1:
                raise RuntimeError(
                    "aten._to_copy not supported with more than one argument currently"
                )
        mps_node = self.create_unary_node(node, mps_graph, MPSView)
        view_shape = get_shape(node)

        mps_node.mpsnode_union.num_dims = len(view_shape)
        mps_node.mpsnode_union.shape = view_shape

        mps_graph.mps_nodes.append(mps_node)
