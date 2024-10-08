# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class DivVisitor(NodeVisitor):
    target = "aten.div.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        # FP32 Div is implemented as output=x/y -> output=x*1/y e.g. MUL(x,RECIPROCAL(y))
        recip = tosa_graph.addIntermediate(
            tosa_shape(inputs[1].shape, inputs[1].dim_order), inputs[1].dtype
        )
        tosa_graph.addOperator(TosaOp.Op().RECIPROCAL, [inputs[1].name], [recip.name])

        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL, [inputs[0].name, recip.name], [output.name], attr
        )
