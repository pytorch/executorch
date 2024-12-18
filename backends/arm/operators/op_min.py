# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, List

import executorch.backends.arm.tosa_quant_utils as tqutils

import serializer.tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import tosa_shape

from serializer.tosa_serializer import TosaOp
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
        is_quant_node: bool,
    ) -> None:
        assert inputs[0].dtype == inputs[1].dtype

        input_qparams = cast(dict[int, tqutils.QuantArgs], node.meta["input_qparams"])
        min_output = output

        if inputs[0].dtype == ts.DType.INT8:
            # insert RESCALEs to int32
            x_scale = input_qparams[0].scale
            x_zp = input_qparams[0].zp

            y_scale = input_qparams[1].scale
            y_zp = input_qparams[1].zp

            assert (
                x_zp == y_zp
            ), "Different zp for inputs, MIN should be quantized with shared quantization!"
            assert (
                x_scale == y_scale
            ), "Different scale for input, MIN should be quantized with shared quantization!"

            operand_inputs, scale_back = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node
            )

            output.shape = tosa_shape(output.shape, output.dim_order)
            min_output = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
        else:
            operand_inputs = inputs

        tosa_graph.addOperator(
            TosaOp.Op().MINIMUM,
            [
                operand_inputs[0].name,
                operand_inputs[1].name,
            ],
            [min_output.name],
        )

        if output.dtype == ts.DType.INT8:
            # insert RESCALE from int32 back to int8
            tqutils.insert_rescale_node_back_to_int8(
                tosa_graph, min_output, scale_back, node
            )
