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
    MPSDataType,
    MPSFull,
    MPSFullLike,
    MPSGraph,
    MPSNode,
)
from executorch.backends.apple.mps.utils.mps_utils import (
    edge_dtype_to_mps_dtype,
    get_input_node,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.sym_util import eval_shape


@register_node_visitor
class ConstantOpVisitor(NodeVisitor):
    target = [
        "aten.full.default",
        "aten.empty.memory_format",
        "aten.scalar_tensor.default",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        if len(node.args) >= 3:
            raise AssertionError("Unexpected number of input parameters")

        if node.target == exir_ops.edge.aten.scalar_tensor.default:
            shape = [1]
        else:
            shape = eval_shape(node.args[0])

        if node.target == exir_ops.edge.aten.full.default:
            fill_value = cast(float, node.args[1])
        elif node.target == exir_ops.edge.aten.empty.memory_format:
            fill_value = 0
        elif node.target == exir_ops.edge.aten.scalar_tensor.default:
            fill_value = cast(float, node.args[0])

        if fill_value == float("-inf"):
            fill_value = "-inf"
        elif fill_value == float("inf"):
            fill_value = "inf"

        dtype = MPSDataType.mps_data_type_float32
        if node.kwargs and "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
            dtype = edge_dtype_to_mps_dtype(node.kwargs["dtype"])

        output_id = self.define_tensor(node, mps_graph)
        mps_graph.mps_nodes.append(
            MPSNode(
                mpsnode_union=MPSFull(
                    output_id=output_id,
                    shape=shape,
                    fill_value=fill_value,
                    dtype=dtype,
                )
            )
        )


@register_node_visitor
class FullLikeVisitor(NodeVisitor):
    target = "aten.full_like.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:

        if len(node.args) < 2:
            raise AssertionError("Full op requires at least size & fill_value args")

        mps_node = self.create_unary_node(node, mps_graph, MPSFullLike)

        mps_node.mpsnode_union.fill_value = cast(float, node.args[1])
        mps_node.mpsnode_union.dtype = self.get_serialized_dtype(
            get_input_node(node, 0)
        )
        if node.kwargs and "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
            mps_node.mpsnode_union.dtype = edge_dtype_to_mps_dtype(node.kwargs["dtype"])
        if len(node.args) >= 3:
            raise AssertionError("Unexpected number of input parameters")

        mps_graph.mps_nodes.append(mps_node)
