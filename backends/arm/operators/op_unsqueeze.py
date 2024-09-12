# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
#  Follows this specification: https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

# pyre-unsafe

import serializer.tosa_serializer as ts
import torch.fx
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class UnsqueezeVisitor(NodeVisitor):
    target = "aten.unsqueeze_copy.default"

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

        dim = inputs[1].number
        shape = inputs[0].shape
        rank = len(shape)

        assert -rank - 1 <= dim < rank + 1
        if dim < 0:
            dim = dim + rank + 1

        new_shape = list(shape)
        new_shape.insert(dim, 1)
        new_shape = tosa_shape(new_shape, output.dim_order)

        attr = ts.TosaSerializerAttribute()
        attr.ReshapeAttribute(new_shape)
        tosa_graph.addOperator(
            TosaOp.Op().RESHAPE, [inputs[0].name], [output.name], attr
        )
