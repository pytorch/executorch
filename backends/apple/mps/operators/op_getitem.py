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
from executorch.backends.apple.mps.utils.mps_utils import get_input_node, get_scalar_val


@register_node_visitor
class GetItemVisitor(NodeVisitor):
    target = "getitem"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        self.tensor_to_id[node] = self.tensor_to_id[get_input_node(node, 0)][
            get_scalar_val(node, 1)
        ]
