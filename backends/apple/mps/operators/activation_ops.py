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
    MPSGELU,
    MPSGraph,
    MPSHardTanh,
    MPSLeakyReLU,
    MPSLogSoftmax,
    MPSReLU,
    MPSSoftmax,
)
from executorch.backends.apple.mps.utils.mps_utils import get_scalar_val
from executorch.exir.dialects._ops import ops as exir_ops


@register_node_visitor
class HardTanhVisitor(NodeVisitor):
    target = "aten.hardtanh.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_unary_node(node, mps_graph, MPSHardTanh)
        mps_node.mpsnode_union.min_value = get_scalar_val(node, 1)
        mps_node.mpsnode_union.max_value = get_scalar_val(node, 2)

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class ReLU_LeakyReLU_GELU_Visitor(NodeVisitor):
    target = ["aten.relu.default", "aten.leaky_relu.default", "aten.gelu.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.activation_ops = {
            exir_ops.edge.aten.relu.default: MPSReLU,
            exir_ops.edge.aten.leaky_relu.default: MPSLeakyReLU,
            exir_ops.edge.aten.gelu.default: MPSGELU,
        }

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        node_type = self.activation_ops[node.target]
        mps_node = self.create_unary_node(node, mps_graph, node_type)

        if node_type is MPSLeakyReLU and len(node.args) == 2:
            mps_node.mpsnode_union.negative_slope = cast(float, node.args[1])
        elif (
            node_type is MPSGELU
            and node.kwargs
            and node.kwargs["approximate"] is not None
        ):
            mps_node.mpsnode_union.approximate = node.kwargs["approximate"]

        mps_graph.mps_nodes.append(mps_node)


@register_node_visitor
class Softmax_LogSoftmax_Visitor(NodeVisitor):
    target = ["aten._softmax.default", "aten._log_softmax.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        node_type = (
            MPSSoftmax
            if node.target == exir_ops.edge.aten._softmax.default
            else MPSLogSoftmax
        )
        mps_node = self.create_unary_node(node, mps_graph, node_type)

        mps_node.mpsnode_union.dim = cast(int, node.args[1])
        mps_node.mpsnode_union.half_to_float = cast(bool, node.args[2])

        mps_graph.mps_nodes.append(mps_node)
