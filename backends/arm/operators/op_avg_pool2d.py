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
from executorch.backends.arm.tosa_utils import transpose_helper
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class AvgPool2dVisitor(NodeVisitor):
    target = "aten.avg_pool2d.default"

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
        input_tensor = inputs[0]
        kernel_size_list = inputs[1].special
        stride_size_list = inputs[2].special
        try:
            pad_size_list = inputs[3].special
        except IndexError:
            pad_size_list = [0, 0, 0, 0]

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size_list,
            stride=stride_size_list,
            pad=pad_size_list,
            input_zp=0,
            output_zp=0,
            accum_dtype=8,
        )  # FP32 accum type

        # Torch's input is [N,C,H,W], TOSA is [N, H, W, C],
        # Transpose to align with TOSA
        NHWC_Order = [0, 2, 3, 1]
        input_transposed = transpose_helper(
            tosa_graph, input_tensor, NHWC_Order, output.dtype
        )

        avg_pool2d_res_shape = [output.shape[i] for i in NHWC_Order]
        avg_pool2d_res = tosa_graph.addIntermediate(avg_pool2d_res_shape, output.dtype)
        tosa_graph.addOperator(
            TosaOp.Op().AVG_POOL2D,
            [input_transposed.name],
            [avg_pool2d_res.name],
            attr,
        )

        # TOSA is [N, H, W, C], Transpose back to Torch's [N, C, H, W]
        NCHW_Order = [0, 3, 1, 2]
        attr_output_transpose = ts.TosaSerializerAttribute()
        attr_output_transpose.TransposeAttribute(NCHW_Order)
        tosa_graph.addOperator(
            TosaOp.Op().TRANSPOSE,
            [avg_pool2d_res.name],
            [output.name],
            attr_output_transpose,
        )
