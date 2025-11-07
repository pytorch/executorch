# Copyright 2025 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg

from torch.fx import Node


@register_node_visitor
class GreaterThanVisitor(NodeVisitor):
    target = "aten.gt.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, inputs, ts)
        validate_valid_dtype(
            self.target,
            inputs,
            [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )
        validate_valid_dtype(self.target, output, ts.DType.BOOL, output.tosa_spec)

        attr = ts.TosaSerializerAttribute()
        attr.GreaterAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.GREATER,
            [inputs[0].name, inputs[1].name],
            [output.name],
            attr,
        )
