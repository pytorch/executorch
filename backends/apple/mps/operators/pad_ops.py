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
    MPSConstantPadND,
    MPSGraph,
)
from executorch.exir.sym_util import eval_shape


@register_node_visitor
class ConstantPadNDVisitor(NodeVisitor):
    target = "aten.constant_pad_nd.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSConstantPadND)

        mps_node.mpsnode_union.pad = eval_shape(cast(torch.SymInt, node.args[1]))
        mps_node.mpsnode_union.value = float(node.args[2])

        mps_graph.mps_nodes.append(mps_node)
