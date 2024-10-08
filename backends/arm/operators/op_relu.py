# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import executorch.backends.arm.tosa_quant_utils as tqutils
import serializer.tosa_serializer as ts
import torch.fx
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class ReluVisitor(NodeVisitor):
    target = "aten.relu.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: list[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        attr = ts.TosaSerializerAttribute()

        clamp_min_fp = 0.0
        clamp_max_fp = 0.0
        clamp_min_qs = 0
        clamp_max_qs = 0
        if is_quant_node:
            out_qargs = tqutils.get_quant_node_args(list(node.users)[0])
            clamp_min_qs = tqutils.quantize_value(0, out_qargs)
            clamp_max_qs = tqutils.quantize_value(float("inf"), out_qargs)

        else:
            clamp_min_fp = 0
            clamp_max_fp = float("inf")

        attr.ClampAttribute(
            tosa_graph.builder,
            clamp_min_qs,
            clamp_max_qs,
            clamp_min_fp,
            clamp_max_fp,
        )

        tosa_graph.addOperator(TosaOp.Op().CLAMP, [inputs[0].name], [output.name], attr)
