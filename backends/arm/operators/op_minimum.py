# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import executorch.backends.arm.tosa_quant_utils as tqutils

import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape
from torch.fx import Node


@register_node_visitor
class MinVisitor(NodeVisitor):
    target = "aten.minimum.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if inputs[0].dtype != inputs[1].dtype and inputs[0].dtype != output.dtype:
            raise TypeError(
                f"Data type of inputs and output must be the same. Got input 0 dtype: "
                f"{inputs[0].dtype}, input 1 dtype: {inputs[1].dtype} and output "
                f"dtype: {output.dtype}"
            )

        scale_back = 1.0
        min_output = output
        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            if len(input_qparams) != 2:
                raise ValueError(
                    f"Both inputs need to have quantization information for {node}"
                )
            if input_qparams[0] != input_qparams[1]:
                raise ValueError(
                    "Both inputs must have the same quantization parameters for MIN"
                )

            # insert RESCALEs to int32
            operand_inputs, scale_back = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node
            )

            output.shape = tosa_shape(output.shape, output.dim_order)
            min_output = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
        else:
            operand_inputs = inputs

        tosa_graph.addOperator(
            ts.TosaOp.Op().MINIMUM,
            [
                operand_inputs[0].name,
                operand_inputs[1].name,
            ],
            [min_output.name],
        )

        if output.dtype == ts.DType.INT8:
            # insert RESCALE from int32 back to int8
            tqutils.insert_rescale_op_to_int8(tosa_graph, min_output, scale_back, node)
