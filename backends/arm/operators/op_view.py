# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
import tosa.Op as TosaOp

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape


@register_node_visitor
class ViewVisitor(NodeVisitor):
    target = "aten.view_copy.default"

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
        attr = ts.TosaSerializerAttribute()
        new_shape = tosa_shape(inputs[1].special, output.dim_order)
        attr.ReshapeAttribute(new_shape)
        tosa_graph.addOperator(
            TosaOp.Op().RESHAPE, [inputs[0].name], [output.name], attr
        )
