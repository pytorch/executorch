#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.backends.apple.mps.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.apple.mps.serialization.mps_graph_schema import (
    MPSAdd,
    MPSBitwiseAnd,
    MPSBitwiseOr,
    MPSBitwiseXor,
    MPSDiv,
    MPSEq,
    MPSFmod,
    MPSGe,
    MPSGraph,
    MPSGt,
    MPSLe,
    MPSLt,
    MPSMinimum,
    MPSMul,
    MPSNe,
    MPSPow,
    MPSRemainder,
    MPSSub,
)
from executorch.exir.dialects._ops import ops as exir_ops


@register_node_visitor
class BinaryOpVisitor(NodeVisitor):
    target = [
        # Arithmetic Binary Ops
        "aten.add.Tensor",
        "aten.add.Scalar",
        "aten.sub.Tensor",
        "aten.sub.Scalar",
        "aten.div.Tensor",
        "aten.div.Tensor_mode",
        "aten.mul.Tensor",
        "aten.mul.Scalar",
        "aten.pow.Tensor_Tensor",
        "aten.pow.Tensor_Scalar",
        "aten.floor_divide.default",
        "aten.fmod.Tensor",
        "aten.fmod.Scalar",
        "aten.remainder.Tensor",
        "aten.remainder.Scalar",
        "aten.bitwise_and.Tensor",
        "aten.bitwise_and.Scalar",
        "aten.bitwise_or.Tensor",
        "aten.bitwise_or.Scalar",
        "aten.bitwise_xor.Tensor",
        "aten.bitwise_xor.Scalar",
        "aten.minimum.default",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.op_mapping = {
            exir_ops.edge.aten.add.Tensor: MPSAdd,
            exir_ops.edge.aten.add.Scalar: MPSAdd,
            exir_ops.edge.aten.sub.Tensor: MPSSub,
            exir_ops.edge.aten.sub.Scalar: MPSSub,
            exir_ops.edge.aten.div.Tensor: MPSDiv,
            exir_ops.edge.aten.div.Tensor_mode: MPSDiv,
            exir_ops.edge.aten.mul.Tensor: MPSMul,
            exir_ops.edge.aten.mul.Scalar: MPSMul,
            exir_ops.edge.aten.pow.Tensor_Tensor: MPSPow,
            exir_ops.edge.aten.pow.Tensor_Scalar: MPSPow,
            exir_ops.edge.aten.floor_divide.default: MPSDiv,
            exir_ops.edge.aten.fmod.Tensor: MPSFmod,
            exir_ops.edge.aten.fmod.Scalar: MPSFmod,
            exir_ops.edge.aten.remainder.Tensor: MPSRemainder,
            exir_ops.edge.aten.remainder.Scalar: MPSRemainder,
            exir_ops.edge.aten.bitwise_and.Tensor: MPSBitwiseAnd,
            exir_ops.edge.aten.bitwise_and.Scalar: MPSBitwiseAnd,
            exir_ops.edge.aten.bitwise_or.Tensor: MPSBitwiseOr,
            exir_ops.edge.aten.bitwise_or.Scalar: MPSBitwiseOr,
            exir_ops.edge.aten.bitwise_xor.Tensor: MPSBitwiseXor,
            exir_ops.edge.aten.bitwise_xor.Scalar: MPSBitwiseXor,
            exir_ops.edge.aten.minimum.default: MPSMinimum,
        }

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_node = self.create_binary_node(
            node, mps_graph, self.op_mapping[node.target]
        )

        if node.kwargs and "alpha" in node.kwargs and node.kwargs["alpha"] is not None:
            mps_node.mpsnode_union.alpha = node.kwargs["alpha"]

        if (
            node.kwargs
            and "rounding_mode" in node.kwargs
            and node.kwargs["rounding_mode"] is not None
        ):
            mps_node.mpsnode_union.rounding_mode = node.kwargs["rounding_mode"]

        mps_graph.mps_nodes.append(mps_node)


##
## Boolean Binary Ops
##
@register_node_visitor
class ComparasionOpVisitor(NodeVisitor):
    target = [
        "aten.eq.Tensor",
        "aten.ne.Tensor",
        "aten.ge.Tensor",
        "aten.gt.Tensor",
        "aten.le.Tensor",
        "aten.lt.Tensor",
        "aten.eq.Scalar",
        "aten.ne.Scalar",
        "aten.ge.Scalar",
        "aten.gt.Scalar",
        "aten.le.Scalar",
        "aten.lt.Scalar",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.comparison_ops = {
            exir_ops.edge.aten.eq.Tensor: MPSEq,
            exir_ops.edge.aten.ne.Tensor: MPSNe,
            exir_ops.edge.aten.ge.Tensor: MPSGe,
            exir_ops.edge.aten.gt.Tensor: MPSGt,
            exir_ops.edge.aten.le.Tensor: MPSLe,
            exir_ops.edge.aten.lt.Tensor: MPSLt,
            exir_ops.edge.aten.eq.Scalar: MPSEq,
            exir_ops.edge.aten.ne.Scalar: MPSNe,
            exir_ops.edge.aten.ge.Scalar: MPSGe,
            exir_ops.edge.aten.gt.Scalar: MPSGt,
            exir_ops.edge.aten.le.Scalar: MPSLe,
            exir_ops.edge.aten.lt.Scalar: MPSLt,
        }

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:

        mps_graph.mps_nodes.append(
            self.create_binary_node(node, mps_graph, self.comparison_ops[node.target])
        )
