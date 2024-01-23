#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import MPSGraph
from executorch.backends.apple.mps.utils.mps_utils import get_input_node
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
        input_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        self.tensor_to_id[node] = input_id
