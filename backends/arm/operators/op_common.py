# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import get_quant_node_args
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

    if is_quant_node:
        # Accumulator type always is int32 when input tensor is an integer type.
        accumulator_type = ts.DType.INT32

    # Initilize zero point to zero.
    input_zp = 0
    output_zp = 0

    if is_quant_node:
        input_zp = get_quant_node_args(node.args[0]).zp
        output_zp = get_quant_node_args(list(node.users)[0]).zp

    attr = ts.TosaSerializerAttribute()
    attr.PoolAttribute(
        kernel=kernel_size,
        stride=stride,
        pad=padding,
        input_zp=input_zp,
        output_zp=output_zp,
        accum_dtype=accumulator_type,
    )

    tosa_graph.addOperator(
        TosaOp.Op().AVG_POOL2D,
        [input_tensor.name],
        [output.name],
        attr,
    )
