# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import get_quant_node_args
from executorch.backends.arm.tosa_utils import transpose_helper
from serializer.tosa_serializer import TosaOp


def build_avg_pool_2d_common(
    node: torch.fx.Node,
    tosa_graph: ts.TosaSerializer,
    input_tensor: TosaArg,
    kernel_size: list,
    stride: list,
    padding: list,
    is_quant_node: bool,
    output: TosaArg,
):
    accumulator_type = input_tensor.dtype
    input_output_type = input_tensor.dtype

    if is_quant_node:
        # Accumulator type always is int32 when input tensor is an integer type.
        accumulator_type = ts.DType.INT32
        input_output_type = ts.DType.INT8

    # Torch's input is [N,C,H,W], TOSA is [N, H, W, C],
    # Transpose to align with TOSA
    NHWC_Order = [0, 2, 3, 1]
    input_transposed = transpose_helper(
        tosa_graph,
        input_tensor,
        NHWC_Order,
        input_output_type,
    )

    # Initilize zero point to zero.
    input_zp = 0
    output_zp = 0

    if is_quant_node:
        _, input_zp = get_quant_node_args(node.args[0])
        _, output_zp = get_quant_node_args(list(node.users)[0])

    attr = ts.TosaSerializerAttribute()
    attr.PoolAttribute(
        kernel=kernel_size,
        stride=stride,
        pad=padding,
        input_zp=input_zp,
        output_zp=output_zp,
        accum_dtype=accumulator_type,
    )

    avg_pool2d_res_shape = [output.shape[i] for i in NHWC_Order]
    avg_pool2d_res = tosa_graph.addIntermediate(avg_pool2d_res_shape, input_output_type)
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
