# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, cast, List

import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape


@register_node_visitor
class ViewVisitor_0_80(NodeVisitor):
    target = "aten.view_copy.default"

    tosa_specs = NodeVisitor.tosa_specs_0_80

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        attr = ts.TosaSerializerAttribute()
        new_shape = tosa_shape(inputs[1].special, output.dim_order)
        attr.ReshapeAttribute(new_shape)
        tosa_graph = cast(ts.TosaSerializer, tosa_graph)
        tosa_graph.addOperator(
            ts.TosaOp.Op().RESHAPE, [inputs[0].name], [output.name], attr
        )


@register_node_visitor
class ViewVisitor(NodeVisitor):
    target = "aten.view_copy.default"

    tosa_specs = NodeVisitor.tosa_specs_1_00

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        tosa_graph = cast(ts.TosaSerializer, tosa_graph)

        if len(output.shape) != 0:
            shape_len = [len(output.shape)]
            shape_data = list(tosa_shape(output.shape, output.dim_order))
        else:
            shape_len = []
            shape_data = []

        shape = tosa_graph.addConst(
            shape_len,
            ts.DType.SHAPE,
            shape_data,
            name=node.name + "_shape",
        )

        attr = ts.TosaSerializerAttribute()
        attr.ReshapeAttribute()
        tosa_graph.addOperator(
            ts.TosaOp.Op().RESHAPE, [inputs[0].name, shape.name], [output.name], attr
        )
