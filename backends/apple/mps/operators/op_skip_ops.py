#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.backends.apple.mps.operators.node_visitor import NodeVisitor
from executorch.backends.apple.mps.serialization.mps_graph_schema import MPSGraph


class OpSkipOps(NodeVisitor):
    """
    Parent Class for handling Skip Ops
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        return
