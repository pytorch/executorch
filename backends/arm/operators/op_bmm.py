# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale
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
    ) -> None:

        assert inputs[0].dtype == inputs[1].dtype, "Both inputs must be of same type"
        assert inputs[0].dtype in [
            ts.DType.INT8,
            ts.DType.FP32,
        ], "Only int8 and float32 supported"
        # aten.bmm maps directly to MATMUL
        # NOTE: For now, only INT8 & FP32 is supported

        # For INT8, we need to get the zero points and add an intermediate tensor
        # for a later rescale.

        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)  # pyre-ingore[16]
            input0_zp = input_qparams[0].zp
            input1_zp = input_qparams[1].zp
            bmm_result = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            bmm_output_name = bmm_result.name
        else:
            bmm_output_name = output.name
            input0_zp, input1_zp = 0, 0

        # Add the MATMUL to the TOSA graph.
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [inputs[0].name, inputs[1].name],
            [bmm_output_name],
            attr,
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        if output.dtype == ts.DType.INT8:
            output_qparams = get_output_qparams(node)[0]  # pyre-ignore[16]
            final_output_scale = (
                input_qparams[0].scale * input_qparams[1].scale  # pyre-ignore[61]
            ) / output_qparams.scale

            build_rescale(
                tosa_fb=tosa_graph,
                scale=final_output_scale,
                # pyre-ignore[61]: Uninitialized local [61]: Local variable `bmm_result` is undefined, or not always defined.
                input_node=bmm_result,
                output_name=output.name,
                output_type=ts.DType.INT8,
                output_shape=bmm_result.shape,
                input_zp=0,
                output_zp=output_qparams.zp,
                is_double_round=False,
            )
