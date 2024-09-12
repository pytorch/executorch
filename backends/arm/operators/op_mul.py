# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts
import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class MulVisitor(NodeVisitor):
    target = "aten.mul.Tensor"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:

        if is_quant_node:
            input_A = inputs[0]
            input_B = inputs[1]
            input_A_qargs = tqutils.get_quant_node_args(
                cast(torch.fx.Node, node.args[0])
            )
            input_B_qargs = tqutils.get_quant_node_args(
                cast(torch.fx.Node, node.args[1])
            )

            input_A.shape = tutils.tosa_shape(input_A.shape, input_A.dim_order)
            input_B.shape = tutils.tosa_shape(input_B.shape, input_B.dim_order)
            output_shape = tutils.tosa_shape(output.shape, output.dim_order)

            # Rescale inputs to INT32 with zp=0
            input_A_rescaled = tqutils.build_rescale_to_int32(
                tosa_graph,
                input_A,
                input_A_qargs.zp,
                rescale_scale=1.0,
            )
            input_B_rescaled = tqutils.build_rescale_to_int32(
                tosa_graph,
                input_B,
                input_B_qargs.zp,
                rescale_scale=1.0,
            )

            mul_output = tosa_graph.addIntermediate(output_shape, ts.DType.INT32)

            # Do the INT32 Mul
            attr = ts.TosaSerializerAttribute()
            attr.MulAttribute(shift=0)
            tosa_graph.addOperator(
                TosaOp.Op().MUL,
                [
                    input_A_rescaled.name,
                    input_B_rescaled.name,
                ],
                [mul_output.name],
                attr,
            )

            tqutils.rescale_node_back_to_int8(
                node, mul_output, input_A_qargs.scale * input_B_qargs.scale, tosa_graph
            )

        else:
            attr = ts.TosaSerializerAttribute()
            attr.MulAttribute(shift=0)
            tosa_graph.addOperator(
                TosaOp.Op().MUL, [inputs[0].name, inputs[1].name], [output.name], attr
            )
