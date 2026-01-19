# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class AbsVisitor(NodeVisitor):
    target = "aten.abs.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [*inputs, output], ts)

        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT32, ts.DType.FP32],
            self.tosa_spec,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AbsAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.ABS,
            [inputs[0].name],
            [output.name],
            attr,
        )
