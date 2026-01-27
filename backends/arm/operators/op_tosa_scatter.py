# Copyright 2026 Arm Limited and/or its affiliates.
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
from torch.fx import Node


@register_node_visitor
class ScatterVisitor(NodeVisitor):
    target = "tosa.SCATTER.default"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], inputs[2], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], inputs[1], inputs[2], output],
            [
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
                ts.DType.FP32,
                ts.DType.FP16,
            ],
            self.tosa_spec,
        )

        attr = ts.TosaSerializerAttribute()
        attr.ScatterAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.SCATTER,
            [inputs[0].name, inputs[1].name, inputs[2].name],
            [output.name],
            attr,
        )
