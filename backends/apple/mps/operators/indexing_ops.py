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
    MPSEmbedding,
    MPSGraph,
    MPSIndexPut,
    MPSIndexSelect,
    MPSIndexTensor,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node
from executorch.backends.transforms import get_shape
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
class IndexTensorVisitor(NodeVisitor):
    target = "aten.index.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSIndexTensor)
        tensors = cast(List[torch.fx.Node], node.args[1])
        for tensor in tensors:
            mps_node.mpsnode_union.indices_id.append(
                self.define_tensor(tensor, mps_graph)
            )

        mps_graph.mps_nodes.append(mps_node)


# [MPS TODO]: Works on a single iteration of llama2, but subsequent tokens
# are wrong when using Index put. Disabling it for now.
@register_node_visitor
class IndexPutVisitor(NodeVisitor):
    # target = "aten.index_put.default"
    target = "disabled"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def infer_sizes(self, a: List[int], b: List[int]):
        dimsA = len(a)
        dimsB = len(b)
        ndim = dimsA if dimsA > dimsB else dimsB
        expandedSizes = [0] * ndim
        for i in range(ndim - 1, -1, -1):
            offset = ndim - 1 - i
            dimA = dimsA - 1 - offset
            dimB = dimsB - 1 - offset
            sizeA = a[dimA] if dimA >= 0 else -1
            sizeB = b[dimB] if dimB >= 0 else -1
            expandedSizes[i] = sizeA if sizeB == -1 else sizeB

        return expandedSizes

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSIndexPut)
        updates_shape = get_shape(node.args[2])
        input_shape = get_shape(node.args[0])
        new_shape = []
        if len(updates_shape) != 1 and len(updates_shape) != len(input_shape):
            new_shape = self.infer_sizes(input_shape, updates_shape)
            mps_node.mpsnode_union.values_shape = new_shape

        tensors = cast(List[torch.fx.Node], node.args[1])
        for tensor in tensors:
            mps_node.mpsnode_union.indices_id.append(
                self.define_tensor(tensor, mps_graph)
            )

        mps_node.mpsnode_union.values_id = self.define_tensor(
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
