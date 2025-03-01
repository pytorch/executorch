# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import executorch.backends.arm.tosa_quant_utils as tqutils
import executorch.backends.arm.tosa_utils as tutils

import serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from serializer.tosa_serializer import TosaOp
from torch.fx import Node


@register_node_visitor
class AddVisitor_080_BI(NodeVisitor):
    target = "aten.add.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        # Specification (0.80) states that input and output types
        # should all be the same
        assert inputs[0].dtype == inputs[1].dtype == output.dtype
        # Handle int8 (quantized) and int32
        assert inputs[0].dtype in [ts.DType.INT8, ts.DType.INT32]

        dim_order = (
            inputs[0].dim_order
            if len(inputs[0].shape) > len(inputs[1].shape)
            else inputs[1].dim_order
        )

        if inputs[0].dtype == ts.DType.INT8:
            rescaled_inputs, scale_back = tqutils.insert_rescale_ops_to_int32(
                tosa_graph, inputs, node
            )
        else:
            # input[0].dtype == ts.DType.INT32
            # Non quantized input, natively support by TOSA.ADD
            rescaled_inputs = inputs

        if output.dtype == ts.DType.INT8:
            broadcasted_shape = tutils.tosa_shape(output.shape, output.dim_order)
            add_output = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)
        else:
            # output.dtype == ts.DType.INT32
            add_output = output

        input1, input2 = tutils.reshape_for_broadcast(
            tosa_graph, rescaled_inputs, dim_order
        )

        # Do the INT32 Add
        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [input1.name, input2.name],
            [add_output.name],
            None,
        )

        if output.dtype == ts.DType.INT8:
            # Scale output back to 8 bit
            # pyre-ignore
            tqutils.insert_rescale_op_to_int8(tosa_graph, add_output, scale_back, node)  # type: ignore[possibly-undefined]


@register_node_visitor
class AddVisitor_080_MI(AddVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        # Specification (0.80) states that input and output types
        # should all be the same
        assert inputs[0].dtype == inputs[1].dtype == output.dtype

        if inputs[0].dtype in [ts.DType.INT8, ts.DType.INT32]:
            # Call the inherited define_node for handling integers
            super().define_node(node, tosa_graph, inputs, output)
        else:
            # FP32 Add lowering
            assert inputs[0].dtype == ts.DType.FP32
            assert output.dtype == ts.DType.FP32

            input1, input2 = tutils.reshape_for_broadcast(tosa_graph, inputs)

            # MI lowering
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [input1.name, input2.name],
                [output.name],
                None,
            )
