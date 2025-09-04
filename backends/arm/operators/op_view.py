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
    validate_valid_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape


@register_node_visitor
class ViewVisitor(NodeVisitor):
    target = "aten.view_copy.default"

    tosa_specs = NodeVisitor.tosa_specs

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
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.INT32, ts.DType.FP32, ts.DType.BOOL],
            output.tosa_spec,
        )

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
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().RESHAPE,
            [inputs[0].name, shape.name],
            [output.name],
            attr,
        )
