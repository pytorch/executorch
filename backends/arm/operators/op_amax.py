# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class MaxVisitor(NodeVisitor):
    target = "aten.amax.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        input = inputs[0]
        dim = inputs[1].number
        keep_dims = inputs[2].number
        if not keep_dims:
            raise RuntimeError(
                "TOSA only supports keepdims == True; Did you run the convert_minmax pass?"
            )

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(input.dim_order.index(dim))

        tosa_graph.addOperator(
            TosaOp.Op().REDUCE_MAX, [input.name], [output.name], attr
        )
