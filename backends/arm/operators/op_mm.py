# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale, get_quant_node_args
from executorch.backends.arm.tosa_utils import (
    build_reshape,
    expand_dims,
    get_two_inputs,
)
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class MMVisitor(NodeVisitor):
    target = "aten.mm.default"

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
        input0, input1 = get_two_inputs(node)

        # For atem.mm, the two inputs are of rank 2
        # For TOSA it needs to be rank 3
        # So they need to be reshaped from (H, W) to (1, H, W)
        # NOTE: For now, only INT8 & FP32 is supported
        reshape_dtype = ts.DType.INT8 if is_quant_node else ts.DType.FP32
        input0_reshaped = expand_dims(tosa_graph, inputs[0], reshape_dtype, 0)
        input1_reshaped = expand_dims(tosa_graph, inputs[1], reshape_dtype, 0)

        # The output also needs to be rank 3
        output_new_shape = (1, output.shape[0], output.shape[1])

        # For INT8, we need to get the zero point, otherwise it is 0
        input0_zp, input1_zp = 0, 0
        if is_quant_node:
            input0_zp = get_quant_node_args(input0).zp
            input1_zp = get_quant_node_args(input1).zp

        mat_mul_result = tosa_graph.addIntermediate(
            output_new_shape, ts.DType.INT32 if is_quant_node else output.dtype
        )

        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input0_reshaped.name, input1_reshaped.name],
            [mat_mul_result.name],
            attr,
        )

        if is_quant_node:
            reshape_intermediate = tosa_graph.addIntermediate(
                output.shape, ts.DType.INT32
            )
            reshape_output_name = reshape_intermediate.name
        else:
            reshape_output_name = output.name

        # Reshape the final output back to rank 2
        build_reshape(
            tosa_graph, mat_mul_result.name, output.shape, reshape_output_name
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        if is_quant_node:
            input0_q_params = get_quant_node_args(input0)
            input1_q_params = get_quant_node_args(input1)
            output_q_params = get_quant_node_args(list(node.users)[0])

            final_output_scale = (
                input0_q_params.scale * input1_q_params.scale
            ) / output_q_params.scale

            # As the input will be INT32, the input_zp must be set to 0
            build_rescale(
                tosa_fb=tosa_graph,
                scale=final_output_scale,
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `reshape_intermediate` is undefined, or not always defined.
                input_node=reshape_intermediate,
                output_name=output.name,
                output_type=ts.DType.INT8,
                output_shape=reshape_intermediate.shape,
                input_zp=0,
                output_zp=output_q_params.zp,
                is_double_round=False,
            )
