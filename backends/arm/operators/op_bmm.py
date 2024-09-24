# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch.fx
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale, get_quant_node_args
from executorch.backends.arm.tosa_utils import get_two_inputs
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class BMMVisitor(NodeVisitor):
    target = "aten.bmm.default"

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

        # aten.bmm maps directly to MATMUL
        # NOTE: For now, only INT8 & FP32 is supported

        # For INT8, we need to get the zero points and add an intermediate tensor
        # for a later rescale.
        if is_quant_node:
            input0_zp = get_quant_node_args(input0).zp
            input1_zp = get_quant_node_args(input1).zp
            bmm_result = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            bmm_output_name = bmm_result.name
        else:
            input0_zp, input1_zp = 0, 0
            bmm_output_name = output.name

        # Add the MATMUL to the TOSA graph.
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input0.name, input1.name],
            [bmm_output_name],
            attr,
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        if is_quant_node:
            input0_q_params = get_quant_node_args(input0)
            input1_q_params = get_quant_node_args(input1)
            output_q_params = get_quant_node_args(list(node.users)[0])

            final_output_scale = (
                input0_q_params.scale * input1_q_params.scale
            ) / output_q_params.scale

            build_rescale(
                tosa_fb=tosa_graph,
                scale=final_output_scale,
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `bmm_result` is undefined, or not always defined.
                input_node=bmm_result,
                output_name=output.name,
                output_type=ts.DType.INT8,
                output_shape=bmm_result.shape,
                input_zp=0,
                output_zp=output_q_params.zp,
                is_double_round=False,
            )
