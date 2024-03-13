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
    MPSEmbedding,
    MPSGraph,
    MPSIndexSelect,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node
from executorch.exir.sym_util import eval_expr


@register_node_visitor
class IndexSelectVisitor(NodeVisitor):
    target = "aten.index_select.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSIndexSelect)
        mps_node.mpsnode_union.dim = cast(int, node.args[1])
        mps_node.mpsnode_union.index_id = self.define_tensor(
            get_input_node(node, 2), mps_graph
        )

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class EmbeddingVisitor(NodeVisitor):
    target = "aten.embedding.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        n_args = len(node.args)
        mps_node = self.create_binary_node(
            node,
            mps_graph,
            MPSEmbedding,
        )

        if n_args >= 3:
            mps_node.mpsnode_union.padding_idx = eval_expr(
                cast(torch.SymInt, node.args[2])
            )
        if n_args >= 4:
            mps_node.mpsnode_union.scale_grad_by_freq = cast(bool, node.args[3])
        if n_args >= 5:
            mps_node.mpsnode_union.sparse = cast(bool, node.args[4])
        mps_graph.mps_nodes.append(mps_node)
