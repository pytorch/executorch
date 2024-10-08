# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

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
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        tensors = inputs[0].special
        dim = 0 if len(inputs) < 2 else inputs[1].number
        rank = len(output.shape)
        dim = (dim + rank) % rank
        dim = output.dim_order.index(dim)

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(dim)

        tosa_graph.addOperator(
            TosaOp.Op().CONCAT, [tensor.name for tensor in tensors], [output.name], attr
        )
