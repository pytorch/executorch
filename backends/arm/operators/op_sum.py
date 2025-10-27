# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class SumVisitor(NodeVisitor):
    target = "aten.sum.dim_IntList"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        tensor = inputs[0]
        input_shape = list(tensor.shape)
        dim = int(inputs[1].number % len(input_shape))

        attr = ts.TosaSerializerAttribute()
        attr.ReduceSumAttribute(tensor.dim_order.index(dim))

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.REDUCE_SUM,
            [tensor.name],
            [output.name],
            attr,
        )
