# Copyright 2024 Arm Limited and/or its affiliates.
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
from executorch.backends.arm.tosa_utils import (
    get_quant_arg_downstream,
    get_quant_arg_upstream,
)

from serializer.tosa_serializer import TosaOp


@register_node_visitor
class MaxPool2dVisitor(NodeVisitor):
    target = "aten.max_pool2d.default"

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
        kernel_size = inputs[1].special
        stride = inputs[2].special

        try:
            padding = [*inputs[3].special, *inputs[3].special]
        except IndexError:
            padding = [0, 0, 0, 0]

        accumulator_type = input_tensor.dtype

        if is_quant_node:
            # Accumulator type always is int8 when input tensor is an integer type.
            accumulator_type = ts.DType.INT8

        # Initilize zero point to zero.
        input_zp = 0
        output_zp = 0

        if is_quant_node:
            input_zp = get_quant_arg_upstream(node.all_input_nodes[0]).zp
            output_zp = get_quant_arg_downstream(list(node.users)[0]).zp

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
            TosaOp.Op().MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )
