# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
)

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class MulVisitor_080_BI(NodeVisitor):
    target = "aten.mul.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        assert inputs[0].dtype == inputs[1].dtype == output.dtype == ts.DType.INT8
        input_A = inputs[0]
        input_B = inputs[1]
        input_qparams = get_input_qparams(node)  # pyre-ignore[16]
        input_A_qargs = input_qparams[0]
        input_B_qargs = input_qparams[1]
        input_A.shape = tutils.tosa_shape(input_A.shape, input_A.dim_order)
        input_B.shape = tutils.tosa_shape(input_B.shape, input_B.dim_order)

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

        output_shape = tutils.tosa_shape(output.shape, output.dim_order)
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
        output_scale = input_A_qargs.scale * input_B_qargs.scale
        tqutils.insert_rescale_op_to_int8(tosa_graph, mul_output, output_scale, node)


@register_node_visitor
class MulVisitor_080_MI(MulVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if inputs[0].dtype == ts.DType.INT8:
            return super().define_node(node, tosa_graph, inputs, output)
        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(shift=0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL, [inputs[0].name, inputs[1].name], [output.name], attr
        )
