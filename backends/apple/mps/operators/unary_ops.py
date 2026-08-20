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
    MPSAbs,
    MPSAcos,
    MPSAcosh,
    MPSAsin,
    MPSAsinh,
    MPSAtan,
    MPSAtanh,
    MPSBitwiseNot,
    MPSCeil,
    MPSCos,
    MPSCosh,
    MPSErf,
    MPSExp,
    MPSExp2,
    MPSFloor,
    MPSGraph,
    MPSIsinf,
    MPSIsnan,
    MPSLog,
    MPSLog10,
    MPSLog2,
    MPSLogicalNot,
    MPSNeg,
    MPSReciprocal,
    MPSRound,
    MPSRsqrt,
    MPSSigmoid,
    MPSSign,
    MPSSin,
    MPSSinh,
    MPSSqrt,
    MPSTan,
    MPSTanh,
)
from executorch.exir.dialects._ops import ops as exir_ops


@register_node_visitor
class UnaryOpVisitor(NodeVisitor):
    target = [
        "aten.exp.default",
        "aten.exp2.default",
        "aten.reciprocal.default",
        "aten.sqrt.default",
        "aten.neg.default",
        "aten.log.default",
        "aten.log10.default",
        "aten.log2.default",
        "aten.erf.default",
        "aten.floor.default",
        "aten.ceil.default",
        "aten.rsqrt.default",
        "aten.sigmoid.default",
        "aten.sin.default",
        "aten.sign.default",
        "aten.cos.default",
        "aten.tan.default",
        "aten.abs.default",
        "aten.asin.default",
        "aten.acos.default",
        "aten.atan.default",
        "aten.sinh.default",
        "aten.cosh.default",
        "aten.tanh.default",
        "aten.asinh.default",
        "aten.acosh.default",
        "aten.atanh.default",
        "aten.bitwise_not.default",
        "aten.isnan.default",
        "aten.isinf.default",
        "aten.round.default",
        "aten.logical_not.default",
    ]

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.unary_op = {
            exir_ops.edge.aten.exp.default: MPSExp,
            exir_ops.edge.aten.exp2.default: MPSExp2,
            exir_ops.edge.aten.reciprocal.default: MPSReciprocal,
            exir_ops.edge.aten.sqrt.default: MPSSqrt,
            exir_ops.edge.aten.neg.default: MPSNeg,
            exir_ops.edge.aten.log.default: MPSLog,
            exir_ops.edge.aten.log10.default: MPSLog10,
            exir_ops.edge.aten.log2.default: MPSLog2,
            exir_ops.edge.aten.erf.default: MPSErf,
            exir_ops.edge.aten.floor.default: MPSFloor,
            exir_ops.edge.aten.ceil.default: MPSCeil,
            exir_ops.edge.aten.rsqrt.default: MPSRsqrt,
            exir_ops.edge.aten.sigmoid.default: MPSSigmoid,
            exir_ops.edge.aten.sin.default: MPSSin,
            exir_ops.edge.aten.sign.default: MPSSign,
            exir_ops.edge.aten.cos.default: MPSCos,
            exir_ops.edge.aten.tan.default: MPSTan,
            exir_ops.edge.aten.abs.default: MPSAbs,
            exir_ops.edge.aten.asin.default: MPSAsin,
            exir_ops.edge.aten.acos.default: MPSAcos,
            exir_ops.edge.aten.atan.default: MPSAtan,
            exir_ops.edge.aten.sinh.default: MPSSinh,
            exir_ops.edge.aten.cosh.default: MPSCosh,
            exir_ops.edge.aten.tanh.default: MPSTanh,
            exir_ops.edge.aten.asinh.default: MPSAsinh,
            exir_ops.edge.aten.acosh.default: MPSAcosh,
            exir_ops.edge.aten.atanh.default: MPSAtanh,
            exir_ops.edge.aten.bitwise_not.default: MPSBitwiseNot,
            exir_ops.edge.aten.isnan.default: MPSIsnan,
            exir_ops.edge.aten.isinf.default: MPSIsinf,
            exir_ops.edge.aten.round.default: MPSRound,
            exir_ops.edge.aten.logical_not.default: MPSLogicalNot,
        }

    def define_node(
        self,
        node: torch.fx.Node,
        mps_graph: MPSGraph,
    ) -> None:
        mps_graph.mps_nodes.append(
            self.create_unary_node(node, mps_graph, self.unary_op[node.target])
        )
