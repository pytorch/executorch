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
    MPSCat,
    MPSExpand,
    MPSGraph,
    MPSNode,
    MPSPermute,
    MPSPixelShuffle,
    MPSSelect,
    MPSSlice,
    MPSSplitWithSizes,
    MPSSqueeze,
    MPSUnsqueeze,
    MPSView,
)
from executorch.backends.apple.mps.utils.mps_utils import get_input_node
from executorch.backends.transforms import get_shape
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.sym_util import eval_expr, eval_shape


@register_node_visitor
class PermuteVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSPermute)

        permute_order = cast(List[int], node.args[1])
        mps_node.mpsnode_union.num_dims = len(permute_order)
        mps_node.mpsnode_union.perm = permute_order

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class ViewExpandVisitor(NodeVisitor):
    target = ["aten.view_copy.default", "aten.expand_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        node_type = (
            MPSView
            if node.target is exir_ops.edge.aten.view_copy.default
            else MPSExpand
        )
        mps_node = self.create_unary_node(node, mps_graph, node_type)

        view_shape = cast(List[int], node.args[1])
        mps_node.mpsnode_union.num_dims = len(view_shape)
        mps_node.mpsnode_union.shape = view_shape

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        tensors = cast(List[torch.fx.Node], node.args[0])
        output_id = self.define_tensor(node, mps_graph)
        input_ids: List[int] = []

        for tensor in tensors:
            input_ids.append(self.define_tensor(tensor, mps_graph))

        dim = 0
        if len(node.args) > 1:
            dim = cast(int, node.args[1])
            if dim < 0 and len(tensors) > 0:
                dim += len(get_shape(tensors[0]))

        mps_graph.mps_nodes.append(
            MPSNode(
                mpsnode_union=MPSCat(input_ids=input_ids, output_id=output_id, dim=dim),
            ),
        )


@register_node_visitor
class SqueezeUnsqueezeVisitor(NodeVisitor):
    target = ["aten.unsqueeze_copy.default", "aten.squeeze_copy.dims"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        node_type = (
            MPSUnsqueeze
            if node.target is exir_ops.edge.aten.unsqueeze_copy.default
            else MPSSqueeze
        )

        mps_node = self.create_unary_node(node, mps_graph, node_type)

        if node_type is MPSUnsqueeze:
            mps_node.mpsnode_union.dim = cast(int, node.args[1])
        else:
            dims = cast(List[int], node.args[1])
            input_shape = get_shape(get_input_node(node, 0))
            new_dims = []
            for dim in dims:
                if input_shape[dim] == 1:
                    new_dims.append(dim)
            mps_node.mpsnode_union.dims = new_dims

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class SelectVisitor(NodeVisitor):
    target = "aten.select_copy.int"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSSelect)
        mps_node.mpsnode_union.dim = cast(int, node.args[1])
        mps_node.mpsnode_union.index = eval_expr(cast(torch.SymInt, node.args[2]))
        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class PixelShuffleVisitor(NodeVisitor):
    target = "aten.pixel_shuffle.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSPixelShuffle)
        mps_node.mpsnode_union.upscale_factor = cast(int, node.args[1])
        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class SliceVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSSlice)

        def maybe_wrap_dim(dim: int, n: int) -> List[int]:
            if dim < 0:
                wrapped_dim = dim + n
                if wrapped_dim < 0:
                    wrapped_dim = 0
                return wrapped_dim
            elif dim > n:
                return n
            return dim

        start = None
        end = None
        if len(node.args) >= 2:
            mps_node.mpsnode_union.dim = cast(int, node.args[1])
        if len(node.args) >= 4:
            end = cast(int, node.args[3])
            start = cast(int, node.args[2])
        if len(node.args) >= 5:
            mps_node.mpsnode_union.step = cast(int, node.args[4])

        input_shape = get_shape(get_input_node(node, 0))
        dim_len = input_shape[
            maybe_wrap_dim(mps_node.mpsnode_union.dim, len(input_shape))
        ]

        start_val = start if start is not None else 0
        end_val = end if end is not None else dim_len

        mps_node.mpsnode_union.start = maybe_wrap_dim(start_val, dim_len)
        mps_node.mpsnode_union.end = maybe_wrap_dim(end_val, dim_len)
        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class SplitWithSizesVisitor(NodeVisitor):
    target = "aten.split_with_sizes_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        input1_id = self.define_tensor(get_input_node(node, 0), mps_graph)
        output_ids = self.define_tensor_list(node, mps_graph)
        split_sizes = eval_shape(cast(torch.SymInt, node.args[1]))
        dim = cast(int, node.args[2])
        input_shape = get_shape(get_input_node(node, 0))

        if dim < 0 or dim >= len(input_shape):
            raise RuntimeError(
                f"split_copy: dim {dim} out of range for input tensor with {len(input_shape)} dimensions"
            )

        mps_node = MPSNode(
            mpsnode_union=MPSSplitWithSizes(
                input1_id=input1_id,
                output_ids=output_ids,
                split_sizes=split_sizes,
                dim=dim,
            )
        )
        mps_graph.mps_nodes.append(mps_node)
