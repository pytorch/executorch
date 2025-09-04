# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import executorch.backends.arm.tosa.quant_utils as tqutils
import executorch.backends.arm.tosa.utils as tutils

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class SumVisitor_INT(NodeVisitor):
    target = "aten.sum.dim_IntList"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        tensor = inputs[0]
        input_shape = list(tensor.shape)
        dim = int(inputs[1].number % len(input_shape))

        output_shape = input_shape
        output_shape[dim] = 1  # Output shape is input shape with dim reduced

        # Rescale input to 32 bit
        rescaled_inputs, scale = tqutils.insert_rescale_ops_to_int32(
            tosa_graph, [tensor], node, self.tosa_spec
        )

        attr = ts.TosaSerializerAttribute()
        attr.ReduceSumAttribute(tensor.dim_order.index(dim))

        intermediate = tosa_graph.addIntermediate(
            tutils.tosa_shape(output_shape, tensor.dim_order),
            dtype=ts.DType.INT32,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().REDUCE_SUM,
            [rescaled_inputs[0].name],
            [intermediate.name],
            attr,
        )

        tqutils.insert_rescale_op_to_int8(
            tosa_graph, intermediate, scale, node, self.tosa_spec
        )


@register_node_visitor
class SumVisitor_FP(SumVisitor_INT):
    # inheriting 'target' from INT class

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+FP")]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 3)
        validate_same_dtype(self.target, [inputs[0], output], ts)

        tensor = inputs[0]
        input_shape = list(tensor.shape)
        dim = int(inputs[1].number % len(input_shape))

        output_shape = input_shape
        output_shape[dim] = 1  # Output shape is input shape with dim reduced

        attr = ts.TosaSerializerAttribute()
        attr.ReduceSumAttribute(tensor.dim_order.index(dim))

        tosa_graph.addOperator(
            ts.TosaOp.Op().REDUCE_SUM,
            [tensor.name],
            [output.name],
            attr,
        )
