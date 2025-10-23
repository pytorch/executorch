# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any

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
from executorch.backends.arm.tosa.utils import tosa_shape


@register_node_visitor
class RepeatVisitor(NodeVisitor):
    target = "aten.repeat.default"

    tosa_specs = NodeVisitor.tosa_specs

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: list[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.INT32, ts.DType.INT16, ts.DType.FP32],
            output.tosa_spec,
        )

        multiples = inputs[1].special

        if len(multiples) == 0:
            raise ValueError(f"Length of multiples argument is 0: {inputs[1]}!")

        multiple_shapes = tosa_graph.addConst(
            (len(multiples),),
            ts.DType.SHAPE,
            list(tosa_shape(multiples, output.dim_order)),
            name=node.name + "_multiples",
        )

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.TILE,
            [inputs[0].name, multiple_shapes.name],
            [output.name],
            attr,
        )
