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
    MPSScatter,
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


@register_node_visitor
class IndexPutVisitor(NodeVisitor):
    target = "aten.index_put.default"

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
class SliceScatterVisitor(NodeVisitor):
    target = "aten.slice_scatter.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.invalid_val = 2**63 - 1

    def maybe_wrap_dim(self, dim: int, n: int) -> List[int]:
        if dim < 0:
            wrapped_dim = dim + n
            if wrapped_dim < 0:
                wrapped_dim = 0
            return wrapped_dim
        elif dim > n:
            return n
        return dim

    def get_exapnded_index(self, idx, shape, dim):
        if idx.dim() == 0:
            return idx.expand(shape)

        dim = self.maybe_wrap_dim(dim, len(shape))

        # setup new_index_shape as [BS, 1, ..., idx_size, ..., 1]
        # to reshape index_
        idx_size = idx.size(0)
        new_index_shape = [1] * len(shape)
        new_index_shape[dim] = idx_size

        # Now apply expand to index_
        index = idx.view(new_index_shape)
        new_index_shape = list(shape)
        new_index_shape[dim] = idx_size
        index = index.expand(new_index_shape)

        return index

    def get_slice_scatter_indices(
        self, dim, start, end, step, input_shape, dtype=torch.int64
    ):
        idx = torch.arange(start, end, step, dtype=dtype)
        return self.get_exapnded_index(idx, input_shape, dim)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSScatter)

        start = None
        end = None
        step = 1

        mps_node.mpsnode_union.src_id = self.define_tensor(
            get_input_node(node, 1), mps_graph
        )
        if len(node.args) >= 3:
            mps_node.mpsnode_union.dim = cast(int, node.args[2])
        if len(node.args) >= 4:
            start = cast(int, node.args[3])
        if len(node.args) >= 5 and node.args[4] != self.invalid_val:
            end = cast(int, node.args[4])
        if len(node.args) >= 6:
            step = cast(int, node.args[5])

        input_shape = get_shape(get_input_node(node, 0))
        dim_len = input_shape[
            self.maybe_wrap_dim(mps_node.mpsnode_union.dim, len(input_shape))
        ]

        start_val = start if start is not None else 0
        end_val = end if end is not None else dim_len

        scatter_indices = self.get_slice_scatter_indices(
            mps_node.mpsnode_union.dim, start_val, end_val, step, input_shape
        )
        mps_node.mpsnode_union.idx_id = self.define_constant(scatter_indices, mps_graph)
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
