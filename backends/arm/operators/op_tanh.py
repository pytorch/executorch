# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import numpy as np

import serializer.tosa_serializer as ts
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
from torch.fx import Node


@register_node_visitor
class TanhVisitor(NodeVisitor):
    target = "aten.tanh.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        assert len(node.all_input_nodes) == 1

        if is_quant_node:
            # Assume quantized input is 8 bit.
            assert len(node.users) == 1

            # Create attribute for 8 bit table lookup.
            input_node = node.all_input_nodes[0]
            in_quantargs = get_quant_arg_upstream(input_node)
            output_node = list(node.users)[0]
            out_quantargs = get_quant_arg_downstream(output_node)

            table = tanh_table_8bit(in_quantargs, out_quantargs)
            table_attr = ts.TosaSerializerAttribute()
            table_attr.TableAttribute(table)

            tosa_graph.addOperator(
                TosaOp.Op().TABLE, [inputs[0].name], [output.name], table_attr
            )
        else:
            tosa_graph.addOperator(TosaOp.Op().TANH, [inputs[0].name], [output.name])


def tanh_table_8bit(in_quantargs: QuantArgs, out_quantargs: QuantArgs):
    """
    Returns a table mapping 256 entries to tanh([qmin,qmax])
    Reference: https://www.mlplatform.org/tosa/tosa_spec.html#_tanh
    """

    def tanh(x):
        # Convert quantized input to floating point tanh input space.
        v = dequantize_value(x, in_quantargs)
        # Compute tanh.
        v = np.exp(-2.0 * v)
        v = (1.0 - v) / (1.0 + v)

        # Convert tanh output back to quantized space.
        return quantize_value(v, out_quantargs)

    return [
        tanh(x)
        for x in np.linspace(in_quantargs.qmin, in_quantargs.qmax, 256, dtype=np.int8)
    ]
