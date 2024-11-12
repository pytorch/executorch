# Copyright 2024 Arm Limited and/or its affiliates.
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
class RsqrtVisitor(NodeVisitor):
    target = "aten.rsqrt.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        if is_quant_node:
            # Assume quantized input is 8 bit.
            # Create attribute for 8 bit table lookup.
            input_node = node.all_input_nodes[0]
            in_quantargs = get_quant_arg_upstream(input_node)
            output_node = list(node.users)[0]
            out_quantargs = get_quant_arg_downstream(output_node)
            table = rsqrt_table_8bit(in_quantargs, out_quantargs)
            table_attr = ts.TosaSerializerAttribute()
            table_attr.TableAttribute(table)
            tosa_graph.addOperator(
                TosaOp.Op().TABLE, [inputs[0].name], [output.name], table_attr
            )
        else:
            tosa_graph.addOperator(TosaOp.Op().RSQRT, [inputs[0].name], [output.name])


def rsqrt_table_8bit(in_quantargs: QuantArgs, out_quantargs: QuantArgs):
    """
    Returns a table mapping 256 entries to rqsrt([qmin,qmax])
    Reference: https://www.mlplatform.org/tosa/tosa_spec.html#_rsqrt
    """

    def rqsrt(x):
        # Convert quantized input to floating point rqsrt input space.
        v = dequantize_value(x, in_quantargs)
        # Compute rqsrt.
        v = 1 / np.sqrt(v)
        # Convert rqsrt output back to quantized space.
        return quantize_value(v, out_quantargs)

    return [
        rqsrt(x)
        for x in np.linspace(in_quantargs.qmin, in_quantargs.qmax, 256, dtype=np.int8)
    ]
