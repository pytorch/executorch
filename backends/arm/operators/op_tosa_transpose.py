# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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


@register_node_visitor
class TransposeVisitor(NodeVisitor):
    """
    This node visitor targets the tosa::TRANSPOSE op defined in the
    TOSA backend dialect. Used when switching between tosa_dim_orders.
    Inserts a TOSA TRANSPOSE.
    """

    target = "tosa.TRANSPOSE.default"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [
                ts.DType.BOOL,
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
                ts.DType.FP16,
                ts.DType.FP32,
            ],
            output.tosa_spec,
        )

        output_rank = len(output.shape)
        perms = [dim % output_rank for dim in inputs[1].special]
        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(perms)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.TRANSPOSE,
            [inputs[0].name],
            [output.name],
            attr,
        )
