# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class CatVisitor(NodeVisitor):
    target = "aten.cat.default"

    tosa_specs = NodeVisitor.tosa_specs

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, [1, 2])

        tensors = inputs[0].special
        dim = 0 if len(inputs) < 2 else inputs[1].number
        rank = len(output.shape)
        dim = (dim + rank) % rank
        dim = output.dim_order.index(dim)

        attr = ts.TosaSerializerAttribute()
        attr.ConcatAttribute(dim)

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().CONCAT,
            [tensor.name for tensor in tensors],
            [output.name],
            attr,
        )
