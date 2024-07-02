# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    build_rescale_from_int32,
    build_rescale_to_int32,
)
from executorch.backends.arm.tosa_utils import broadcast_shapes, getNodeArgs, tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class AddVisitor(NodeVisitor):
    target = "aten.add.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        if is_quant_node:
            # Single input or not
            if len(node.all_input_nodes) == 1:
                input_node_A = node.all_input_nodes[0]
                input_node_B = node.all_input_nodes[0]
            else:
                input_node_A, input_node_B = node.all_input_nodes

            # Get input scale_factor and zero_points for A, B
            input_A, input_A_scale, input_A_zp, _, _, _ = getNodeArgs(input_node_A)
            input_B, input_B_scale, input_B_zp, _, _, _ = getNodeArgs(input_node_B)

            # Scale the int8 quantized input to a common scale in the integer
            # domain.
            min_scale = min(input_A_scale.number, input_B_scale.number)
            inputA_rescale_scale = input_A_scale.number / min_scale
            inputB_rescale_scale = input_B_scale.number / min_scale

            input_A.shape = tosa_shape(input_A.shape, input_A.dim_order)
            input_B.shape = tosa_shape(input_B.shape, input_B.dim_order)
            broadcasted_shape = broadcast_shapes(input_A.shape, input_B.shape)

            input_A_rescaled_to_int32 = build_rescale_to_int32(
                tosa_graph,
                input_A,
                input_A_zp.number,
                inputA_rescale_scale,
            )

            input_B_rescaled_to_int32 = build_rescale_to_int32(
                tosa_graph,
                input_B,
                input_B_zp.number,
                inputB_rescale_scale,
            )

            ## Do the INT32 Add
            add_res = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [
                    input_A_rescaled_to_int32.name,
                    input_B_rescaled_to_int32.name,
                ],
                [add_res.name],
                None,
            )

            # Output
            output_node = list(node.users)[0]
            _, output_scale, output_zp, _, _, _ = getNodeArgs(output_node)
            output_rescale_scale = min_scale / output_scale.number

            # Rescale Back to INT8
            build_rescale_from_int32(
                tosa_graph,
                add_res.name,
                output.name,
                output_zp.number,
                output_rescale_scale,
            )
        else:
            # FP32 Add lowering
            tosa_graph.addOperator(
                TosaOp.Op().ADD, [inputs[0].name, inputs[1].name], [output.name], None
            )
