# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import numpy as np

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import (
    dequantize_value,
    get_quant_arg_downstream,
    get_quant_arg_upstream,
    QuantArgs,
    quantize_value,
)
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class DivVisitor(NodeVisitor):
    target = "aten.reciprocal.default"

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
        # 1/X

        if is_quant_node:
            input = inputs[0]
            input_qargs = get_quant_arg_upstream(node.all_input_nodes[0])
            output_qargs = get_quant_arg_downstream(list(node.users)[0])

            div_table = div_table_8bit(input_qargs, output_qargs)

            table_attr = ts.TosaSerializerAttribute()
            table_attr.TableAttribute(div_table)
            tosa_graph.addOperator(
                TosaOp.Op().TABLE, [input.name], [output.name], table_attr
            )

        else:
            tosa_graph.addOperator(
                TosaOp.Op().RECIPROCAL, [inputs[0].name], [output.name]
            )


def div_table_8bit(in_quantargs: QuantArgs, out_quantargs: QuantArgs):
    """
    Returns a table mapping 256 entries to div([qmin,qmax])
    """

    def div(x):
        # Convert quantized input to floating point div input space.
        v1 = dequantize_value(x, in_quantargs)
        # Compute div.
        v2 = 1.0 / v1
        # Convert div output back to quantized space.
        v3 = quantize_value(v2, out_quantargs)

        return v3

    return [
        div(x)
        for x in np.linspace(in_quantargs.qmin, in_quantargs.qmax, 256, dtype=np.int8)
    ]
