# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provide a visitor for lowering batched matmul (BMM) to TOSA."""

from typing import Any, List

import torch
import tosa_serializer as ts

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
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


@register_node_visitor
class MatmulVisitor(NodeVisitor):
    """Provide a visitor that serializes TOSA ``MATMUL``."""

    target = "tosa.MATMUL.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        """Define the TOSA ``MATMUL`` operator."""
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs], ts)
        validate_valid_dtype(
            self.target,
            [*inputs],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.FP32],
            output.tosa_spec,
        )
        validate_valid_dtype(
            self.target,
            [output],
            [ts.DType.INT32, ts.DType.INT48, ts.DType.FP32],
            output.tosa_spec,
        )

        # We need to get the zero points and add an intermediate tensor for INT16 case
        if inputs[0].dtype in (ts.DType.INT8, ts.DType.INT16):
            input_qparams = get_input_qparams(node)
            input0_zp = input_qparams[0].get_zp_per_tensor()
            input1_zp = input_qparams[1].get_zp_per_tensor()
        else:
            input0_zp, input1_zp = 0, 0

        input_A_ZP_name = f"{node.name}_A_ZP"
        input_B_ZP_name = f"{node.name}_B_ZP"
        tosa_graph.addConst([1], inputs[0].dtype, [input0_zp], name=input_A_ZP_name)
        tosa_graph.addConst([1], inputs[1].dtype, [input1_zp], name=input_B_ZP_name)

        # Add the MATMUL to the TOSA graph.
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MATMUL,
            [
                inputs[0].name,
                inputs[1].name,
                input_A_ZP_name,
                input_B_ZP_name,
            ],
            [output.name],
            attr,
        )
