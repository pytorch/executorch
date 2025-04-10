# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape


@register_node_visitor
class RepeatVisitor(NodeVisitor):
    target = "aten.repeat.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: list[TosaArg],
        output: TosaArg,
    ) -> None:

        multiples = inputs[1].special

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute(tosa_shape(multiples, output.dim_order))
        tosa_graph.addOperator(
            ts.TosaOp.Op().TILE, [inputs[0].name], [output.name], attr
        )
