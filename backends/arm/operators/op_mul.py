# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import torch

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


@register_node_visitor
class MulVisitor(NodeVisitor):
    target = "aten.mul.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )

        tosa_graph.addConst([1], ts.DType.INT8, 0, name=f"{node.name}_shift")
        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MUL,
            [inputs[0].name, inputs[1].name, f"{node.name}_shift"],
            [output.name],
            attr,
        )
