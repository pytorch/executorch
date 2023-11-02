# Copyright 2023 Arm Limited and/or its affiliates.
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
    computeMultiplierAndShift,
    getQuantNodeArgs,
)
from executorch.backends.arm.tosa_utils import promote_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class AddmmVisitor(NodeVisitor):
    target = "aten.addmm.default"

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
        bias, input, weight = inputs

        output_dtype = ts.DType.INT8 if is_quant_node else output.dtype

        # Reshape input, weight, bias tensors
        input_reshape_res = promote_shape(
            tosa_graph, input, (1,) + input.shape, output_dtype
        )
        weight_reshape_res = promote_shape(
            tosa_graph, weight, (1,) + weight.shape, output_dtype
        )

        bias_dtype = ts.DType.INT32 if is_quant_node else output.dtype
        bias_reshape_res = promote_shape(
            tosa_graph,
            bias,
            (
                1,
                1,
            )
            + bias.shape,
            bias_dtype,
        )

        # Add dummy batch 1 to mm_shape
        mm_shape = (1, input.shape[0], weight.shape[1])
        # Define Intermediate tensor for MatMul res
        mm_res = tosa_graph.addIntermediate(
            mm_shape, ts.DType.INT32 if is_quant_node else output_dtype
        )

        # Add MatMulOp
        attr_matmul = ts.TosaSerializerAttribute()
        a_zp, b_zp = (-128, 0) if is_quant_node else (0, 0)
        attr_matmul.MatMulAttribute(a_zp, b_zp)
        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input_reshape_res.name, weight_reshape_res.name],
            [mm_res.name],
            attr_matmul,
        )

        # Add AddOp
        add_res = tosa_graph.addIntermediate(
            mm_shape, ts.DType.INT32 if is_quant_node else output_dtype
        )

        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [bias_reshape_res.name, mm_res.name],
            [add_res.name],
            None,
        )

        if is_quant_node:
            # Read inputs' parent nodes
            #
            _, input_node, weight_node = node.all_input_nodes
            input_scale, _ = getQuantNodeArgs(input_node)
            weight_node_q_node = weight_node.all_input_nodes[0]
            weight_scale, _ = getQuantNodeArgs(weight_node_q_node)

            consumer_node = list(node.users)[0]
            consumer_node_scale, consumer_node_node_zp = getQuantNodeArgs(consumer_node)

            output_rescale_scale = (input_scale * weight_scale) / consumer_node_scale
            (
                multiplier_output,
                shift_output,
            ) = computeMultiplierAndShift(output_rescale_scale)

            attr_rescale_output = ts.TosaSerializerAttribute()
            attr_rescale_output.RescaleAttribute(
                input_zp=0,
                output_zp=consumer_node_node_zp,
                multiplier=[multiplier_output],
                shift=[shift_output],
                scale32=True,
                double_round=True,
                per_channel=False,
            )
            add_res_int8 = tosa_graph.addIntermediate(mm_shape, ts.DType.INT8)
            tosa_graph.addOperator(
                TosaOp.Op().RESCALE,
                [add_res.name],
                [add_res_int8.name],
                attr_rescale_output,
            )
        # Reshape final result to original shape
        attr_out = ts.TosaSerializerAttribute()
        attr_out.ReshapeAttribute(output.shape)
        tosa_graph.addOperator(
            TosaOp.Op().RESHAPE,
            [add_res_int8.name if is_quant_node else add_res.name],
            [output.name],
            attr_out,
        )
